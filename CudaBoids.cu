/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Apr 5, 2022
*******************************************************************/

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <list>
#include <random>

/// Requires GLEW (static), GLFW (static), and CUDA (dynamic).
///
/// Commands:
///  Space            -   toggle traces
///  P                -   pause
///  R                -   reset boids
///  T                -   toggle attraction/repulsion
///  ESC              -   quit
///  1 mouse button   -   weak attraction/repulsion
///  2 mouse buttons  -   strong attracton/repulsion

/// configuration
static constexpr bool kFullScreen = true;
static constexpr bool kVsync = false;
static constexpr int kWindowWidth = 800;
static constexpr int kWindowHeight = 450;
static constexpr int kCudaDeviceId = 0;

static constexpr int kNumBoids = 5888;
static constexpr float kNeighborDist = 130.0f;
static constexpr float kBoidLen = 6.0f;
static constexpr float kInitialVel = 10.0f;
static constexpr float kMaxVel = 28.0f;
static constexpr float kAlignFactor = 0.55f;
static constexpr float kRepelFactor = 25.6f;
static constexpr float kCmFactor = 0.86f;
static constexpr float kMouseFactor = 90.0f;
static constexpr float kTwoMouseButtonsFactor = 1000.0f;
static constexpr float kVelScalar = 6.0f;

/// for internal use
static constexpr int kSharedCollateCount = 1;
static constexpr int kCoopWidth = 4;
static constexpr int kThreadsPerBlock = 256;
static constexpr int kPaddedNumBoids = int(uint32_t(kNumBoids + kSharedCollateCount * kThreadsPerBlock - 1) & ~uint32_t(kSharedCollateCount * kThreadsPerBlock - 1));
static constexpr int kNumBlocks = (kNumBoids - 1) / (kThreadsPerBlock / kCoopWidth) + 1;
static constexpr int kFrameHistorySize = 6;

static bool paused = false;
static bool traces = false;
static bool reset = false;
static std::chrono::high_resolution_clock::time_point frameClock;
static float repelMultiplier = 1.0f;
static float mouseX = 0;
static float mouseY = 0;
static int buttonsDown = 0;

#define CHECK_CUDA_ERRORS(x)                                                                                        \
do {                                                                                                                \
    const cudaError_t s = (x);                                                                                      \
    if (s != cudaSuccess)                                                                                           \
    {                                                                                                               \
        printf("CUDA ERROR during " #x "\n%s: %s\n", cudaGetErrorName(s), cudaGetErrorString(s));                   \
        return EXIT_FAILURE;                                                                                        \
    }                                                                                                               \
}                                                                                                                   \
while (0)

#define CHECK_GL_ERRORS(x)                                                                                          \
do {                                                                                                                \
    (x);                                                                                                            \
    {                                                                                                               \
        GLenum err = glGetError();                                                                                  \
        if (err != GL_NO_ERROR)                                                                                     \
        {                                                                                                           \
            do                                                                                                      \
            {                                                                                                       \
                printf("OpenGL ERROR during " #x ": %u\n", err);                                                    \
                err = glGetError();                                                                                 \
            } while (err != GL_NO_ERROR);                                                                           \
            return EXIT_FAILURE;                                                                                    \
        }                                                                                                           \
    }                                                                                                               \
}                                                                                                                   \
while (0)

static void KeyboardCb(GLFWwindow* pWindow, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(pWindow, GL_TRUE);
            break;
        case GLFW_KEY_P:
            paused = !paused;
            frameClock = std::chrono::high_resolution_clock::now();
            break;
        case GLFW_KEY_R:
            reset = true;
            break;
        case GLFW_KEY_T:
            repelMultiplier *= -1.0f;
            break;
        case GLFW_KEY_SPACE:
            traces = !traces;
            break;
        }
    }
}

static void MouseMoveCb(GLFWwindow* /*pWindow*/, double x, double y)
{
    mouseX = float(x);
    mouseY = float(y);
}

static void MouseClickCb(GLFWwindow* pWindow, int /*button*/, int action, int /*mods*/)
{
    buttonsDown += (action == GLFW_PRESS) - (action == GLFW_RELEASE);
    double x, y;
    glfwGetCursorPos(pWindow, &x, &y);
    mouseX = float(x);
    mouseY = float(y);
    glfwSetCursorPosCallback(pWindow, buttonsDown ? MouseMoveCb : nullptr);
}

static void WindowMoveCb(GLFWwindow* /*pWindow*/, int /*x*/, int /*y*/)
{
    frameClock = std::chrono::high_resolution_clock::now();
}

static int Reset(int width, int height, float4* __restrict d_boids)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDis(0.0f, float(width - 1));
    std::uniform_real_distribution<float> yDis(0.0f, float(height - 1));
    std::uniform_real_distribution<float> velDis(-kInitialVel, kInitialVel);
    float4* h_boids = new float4[kPaddedNumBoids];
    for (int i = 0; i < kNumBoids; ++i)
    {
        h_boids[i].x = xDis(gen);
        h_boids[i].y = yDis(gen);
        h_boids[i].z = velDis(gen);
        h_boids[i].w = velDis(gen);
    }
    for (int i = kNumBoids; i < kPaddedNumBoids; ++i)
    {
        h_boids[i].x = -10000.0f;
        h_boids[i].y = -10000.0f;
        h_boids[i].z = 0.0f;
        h_boids[i].w = 0.0f;
    }
    CHECK_CUDA_ERRORS(cudaMemcpy(d_boids, h_boids, kPaddedNumBoids * sizeof(float4), cudaMemcpyHostToDevice));
    delete[] h_boids;
    reset = false;
    return 0;
}

__device__ float Copysign(float a, float b)
{
    float r;
    asm("lop3.b32 %0, %1, 0x80000000U, %2, 0xE2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float Dist(const float c1, const float c2, const float m)
{
    const float d = c2 - c1;
    const float w = d - Copysign(d, m);
    return abs(d) < abs(w) ? d : w;
}

__device__ float Rcp(const float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ float Div(const float n, const float d)
{
    return n * Rcp(d);
}

__device__ float Rem(const float n, const float d)
{
    return fmaf(-d, floorf(Div(n, d)), n);
}

__device__ float Rsqrt(const float x)
{
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ void Reduce(float& x)
{
    uint32_t i = 1;
#pragma unroll
    while (i < kCoopWidth)
    {
        x += __shfl_xor_sync(uint32_t(-1), x, i);
        i <<= 1;
    }
}

static __global__ void BoidsKernel(
    uchar4* __restrict__ const d_pbo,
    const float4* __restrict__ const d_in,
    float4* __restrict__ const d_out,
    const float mouseX,
    const float mouseY,
    const float width,
    const float height,
    const float repelMultiplier,
    const float dt)
{
    const int iBoid = (kThreadsPerBlock / kCoopWidth) * blockIdx.x + threadIdx.y;
    float4 b = d_in[iBoid];

    float numNeighbors = 0.0f;
    float cmX = 0.0f, cmY = 0.0f, repX = 0.0f, repY = 0.0f, alX = 0.0f, alY = 0.0f;

    __shared__ float4 others[kSharedCollateCount * kThreadsPerBlock];

#pragma unroll 128
    for (int iOther = 0; iOther < kPaddedNumBoids / kCoopWidth; iOther += kSharedCollateCount * kThreadsPerBlock / kCoopWidth)
    {
        __syncthreads();
#pragma unroll
        for (int jOther = 0; jOther < kSharedCollateCount; ++jOther)
            others[jOther * kThreadsPerBlock + kCoopWidth * threadIdx.y + threadIdx.x] = d_in[iOther + (kPaddedNumBoids / kCoopWidth) * threadIdx.x + (jOther * kThreadsPerBlock / kCoopWidth) + threadIdx.y];
        __syncthreads();

#pragma unroll 8
        for (int jOther = 0; jOther < kSharedCollateCount * kThreadsPerBlock / kCoopWidth; ++jOther)
        {
            const float4 other = others[threadIdx.x + kCoopWidth * jOther];
            const float dx = Dist(b.x, other.x, width);
            const float dy = Dist(b.y, other.y, height);
            const float lenDSqr = fmaf(dx, dx, fmaf(dy, dy, 0.000000001f));
            const bool pred = lenDSqr < kNeighborDist * kNeighborDist;
            const float invLen = Rcp(lenDSqr);
            if (pred) cmX = fmaf(kCmFactor, dx, cmX);
            if (pred) repX = fmaf(-invLen, dx, repX);
            if (pred) alX = fmaf(kAlignFactor, other.z, alX);
            if (pred) cmY = fmaf(kCmFactor, dy, cmY);
            if (pred) repY = fmaf(-invLen, dy, repY);
            if (pred) alY = fmaf(kAlignFactor, other.w, alY);
            if (pred) numNeighbors += 1.0f;
        }
    }

    if (iBoid >= kNumBoids)
        return;

    if (uint32_t(iBoid - (kPaddedNumBoids / kCoopWidth) * threadIdx.x) < uint32_t(kPaddedNumBoids / kCoopWidth))
    {
        alX = fmaf(-kAlignFactor, b.z, alX);
        alY = fmaf(-kAlignFactor, b.w, alY);
    }

    cmX += alX;
    cmY += alY;

    Reduce(repX);
    Reduce(cmX);

    Reduce(repY);
    Reduce(cmY);

    Reduce(numNeighbors);

    if (threadIdx.x)
        return;

    {
        const float dx = Dist(b.x, mouseX, width);
        const float dy = Dist(b.y, mouseY, height);
        const float lenDSqr = dt * repelMultiplier * Rsqrt(fmaf(dx, dx, fmaf(dy, dy, 0.000000001f)));
        b.z = fmaf(lenDSqr, dx, b.z);
        b.w = fmaf(lenDSqr, dy, b.w);
    }

    b.z = fmaf(dt, fmaf(kRepelFactor, repX, Div(cmX, numNeighbors)), b.z);
    b.w = fmaf(dt, fmaf(kRepelFactor, repY, Div(cmY, numNeighbors)), b.w);

    {
        const float lenVSqr = fmaf(b.z, b.z, b.w * b.w);
        const float vScale = lenVSqr > kMaxVel * kMaxVel ? kMaxVel * Rsqrt(lenVSqr) : 1.0f;
        b.z *= vScale;
        b.w *= vScale;
    }

    b.x = Rem(fmaf(kVelScalar * dt, b.z, b.x), width);
    b.y = Rem(fmaf(kVelScalar * dt, b.w, b.y), height);
    d_out[iBoid] = b;

    {
        const float lenVSqr = Rsqrt(fmaf(b.z, b.z, fmaf(b.w, b.w, 0.000000001f)));
        b.z *= lenVSqr;
        b.w *= lenVSqr;
    }

    const float c = atan2f(b.w, b.z) * (6.0f / 6.28318530718f);
    const uchar4 rgb
    {
        static_cast<unsigned char>(255.0f * __saturatef(abs(Rem(c + 0.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(255.0f * __saturatef(abs(Rem(c + 4.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(255.0f * __saturatef(abs(Rem(c + 2.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(0)
    };

    int x = int(fmaf(-kBoidLen, b.z, b.x));
    const int x2 = int(fmaf(kBoidLen, b.z, b.x));
    int y = int(fmaf(-kBoidLen, b.w, b.y));
    const int y2 = int(fmaf(kBoidLen, b.w, b.y));

    int dx = abs(x2 - x);
    int dy = -abs(y2 - y);
    const int sx = x2 > x ? 1 : -1;
    const int sy = y2 > y ? 1 : -1;
    int err = dx + dy;
    dx <<= 1;
    dy <<= 1;
    const int w = int(width);
    const int h = int(height);
    d_pbo[(y + h * ((y < 0) - (y >= h))) * w + x + w * ((x < 0) - (x >= w))] = rgb;
    do
    {
        x += err > 0 ? sx : 0;
        y += err > 0 ? 0 : sy;
        err += err > 0 ? dy : dx;
        d_pbo[(y + h * ((y < 0) - (y >= h))) * w + x + w * ((x < 0) - (x >= w))] = rgb;
    } while (x != x2 || y != y2);
}

int main()
{
    if (!glfwInit())
    {
        printf("ERROR: Failed to initialize GLFW.\n");
        return EXIT_FAILURE;
    }

    GLFWwindow* pWindow = nullptr;
    if (kFullScreen)
    {
        if (GLFWmonitor* const pPrimary = glfwGetPrimaryMonitor())
        {
            if (const GLFWvidmode* const pMode = glfwGetVideoMode(pPrimary))
            {
                glfwWindowHint(GLFW_RED_BITS, pMode->redBits);
                glfwWindowHint(GLFW_GREEN_BITS, pMode->greenBits);
                glfwWindowHint(GLFW_BLUE_BITS, pMode->blueBits);
                glfwWindowHint(GLFW_REFRESH_RATE, pMode->refreshRate);
                glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
                pWindow = glfwCreateWindow(pMode->width, pMode->height, "Boids", nullptr, nullptr);
            }
        }
    }
    else
    {
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
        pWindow = glfwCreateWindow(kWindowWidth, kWindowHeight, "Boids", nullptr, nullptr);
    }

    if (!pWindow)
    {
        printf("ERROR: Failed to create window.\n");
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(pWindow);
    glfwSwapInterval(kVsync ? 1 : 0);
    int width, height;
    glfwGetFramebufferSize(pWindow, &width, &height);
    glfwSetKeyCallback(pWindow, KeyboardCb);
    glfwSetMouseButtonCallback(pWindow, MouseClickCb);
    glfwSetWindowPosCallback(pWindow, WindowMoveCb);
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0"))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.\n");
        return EXIT_FAILURE;
    }

    CHECK_GL_ERRORS(glViewport(0, 0, width, height));
    CHECK_GL_ERRORS(glClearColor(0.0, 0.0, 0.0, 1.0));
    CHECK_GL_ERRORS(glDisable(GL_DEPTH_TEST));
    CHECK_GL_ERRORS(glMatrixMode(GL_MODELVIEW));
    CHECK_GL_ERRORS(glLoadIdentity());
    CHECK_GL_ERRORS(glMatrixMode(GL_PROJECTION));
    CHECK_GL_ERRORS(glLoadIdentity());
    CHECK_GL_ERRORS(glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f));
    GLuint pbo = 0;
    CHECK_GL_ERRORS(glGenBuffers(1, &pbo));
    CHECK_GL_ERRORS(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));
    CHECK_GL_ERRORS(glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4) * width * height, nullptr, GL_DYNAMIC_DRAW));
    CHECK_GL_ERRORS(glEnable(GL_TEXTURE_2D));
    GLuint textureId = 0;
    CHECK_GL_ERRORS(glGenTextures(1, &textureId));
    CHECK_GL_ERRORS(glBindTexture(GL_TEXTURE_2D, textureId));
    CHECK_GL_ERRORS(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    CHECK_GL_ERRORS(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    CHECK_GL_ERRORS(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr));
    CHECK_CUDA_ERRORS(cudaSetDevice(kCudaDeviceId));
    cudaGraphicsResource_t cgr;
    CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&cgr, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaStream_t cs;
    CHECK_CUDA_ERRORS(cudaStreamCreate(&cs));
    float4* d_boids = nullptr;
    CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_boids), 2 * kPaddedNumBoids * sizeof(float4)));
    if (Reset(width, height, d_boids))
        return EXIT_FAILURE;
    CHECK_CUDA_ERRORS(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    std::chrono::high_resolution_clock::time_point fpsClock = std::chrono::high_resolution_clock::now();
    frameClock = std::chrono::high_resolution_clock::now();
    std::list<std::chrono::nanoseconds> frameTimes{ std::chrono::nanoseconds(0) };
    int fpsFrames = 0;
    std::chrono::nanoseconds frameTimeSum(0);
    bool pingPong = false;

    while (!glfwWindowShouldClose(pWindow))
    {
        if (!paused)
        {
            if (reset && Reset(width, height, pingPong ? d_boids + kPaddedNumBoids : d_boids))
                return EXIT_FAILURE;
            CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &cgr, cs));
            size_t bufSize = 0;
            uchar4* d_pbo = nullptr;
            CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pbo), &bufSize, cgr));

            if (!traces)
                cudaMemset(d_pbo, 0, sizeof(uchar4) * width * height);

            BoidsKernel<<<kNumBlocks, { kCoopWidth, kThreadsPerBlock / kCoopWidth }>>>(
                d_pbo,
                pingPong ? d_boids + kPaddedNumBoids : d_boids,
                pingPong ? d_boids : d_boids + kPaddedNumBoids,
                mouseX,
                mouseY,
                float(width),
                float(height),
                buttonsDown ? repelMultiplier * (buttonsDown > 1 ? kTwoMouseButtonsFactor : kMouseFactor) : 0.0f,
                1e-9f * frameTimeSum.count() / frameTimes.size()
            );

            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &cgr, cs));
            pingPong = !pingPong;
        }
        CHECK_GL_ERRORS(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
        CHECK_GL_ERRORS(glEnd());
        CHECK_GL_ERRORS(glfwSwapBuffers(pWindow));
        CHECK_GL_ERRORS(glfwPollEvents());
        if (frameTimes.size() >= kFrameHistorySize)
        {
            frameTimeSum -= frameTimes.front();
            frameTimes.pop_front();
        }
        const std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        frameTimes.emplace_back(now - frameClock);
        frameTimeSum += frameTimes.back();
        frameClock = now;
        if (now - fpsClock > std::chrono::milliseconds(500))
        {
            printf("FPS: %f\n", double(fpsFrames) / (1e-9 * (now - fpsClock).count()));
            fpsClock = now;
            fpsFrames = 0;
        }
        ++fpsFrames;
    }
}
