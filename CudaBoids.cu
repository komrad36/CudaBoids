/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Apr 2, 2022
*******************************************************************/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <list>
#include <random>

/// configuration
static constexpr bool kFullScreen = true;
static constexpr bool kVsync = false;
static constexpr int kWindowWidth = 800;
static constexpr int kWindowHeight = 450;
static constexpr int kCudaDeviceId = 0;

static constexpr int kNumBoids = 5850;
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
static constexpr int kThreadsPerBlock = 128;
static constexpr int kNumBlocks = ((kNumBoids - 1) / kThreadsPerBlock + 1);
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

static void mouseMoveCallback(GLFWwindow* /*pWindow*/, double x, double y)
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
    glfwSetCursorPosCallback(pWindow, buttonsDown ? mouseMoveCallback : nullptr);
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
    float4* h_boids = new float4[kNumBoids];
    for (int i = 0; i < kNumBoids; ++i)
    {
        h_boids[i].x = xDis(gen);
        h_boids[i].y = yDis(gen);
        h_boids[i].z = velDis(gen);
        h_boids[i].w = velDis(gen);
    }
    CHECK_CUDA_ERRORS(cudaMemcpy(d_boids, h_boids, kNumBoids * sizeof(float4), cudaMemcpyHostToDevice));
    delete[] h_boids;
    reset = false;
    return 0;
}

__device__ float Dist(const float c1, const float c2, const float m)
{
    const float d = c2 - c1;
    const float s = abs(d);
    const float w = d - copysign(m, d);
    return s < 0.5f * m ? d : w;
}

__device__ float Rem(const float n, const float d)
{
    const float r = fmodf(n, d);
    return r < 0.0f ? r + d : r;
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
    const int iBoid = blockDim.x * blockIdx.x + threadIdx.x;
    if (iBoid >= kNumBoids)
        return;

    float4 b = d_in[iBoid];

    int numNeighbors = 0;
    float cmX = 0.0f, cmY = 0.0f, repX = 0.0f, repY = 0.0f, alX = -b.z, alY = -b.w;
    {
        const float dx = Dist(b.x, mouseX, width);
        const float dy = Dist(b.y, mouseY, height);
        const float lenDSqr = dt * repelMultiplier / sqrtf(fmaf(dx, dx, fmaf(dy, dy, 0.000000001f)));
        b.z += dx * lenDSqr;
        b.w += dy * lenDSqr;
    }

#pragma unroll 6
    for (int iOther = 0; iOther < kNumBoids; ++iOther)
    {
        const float dx = Dist(b.x, d_in[iOther].x, width);
        const float dy = Dist(b.y, d_in[iOther].y, height);
        const float lenDSqr = fmaf(dx, dx, fmaf(dy, dy, 0.000000001f));
        if (lenDSqr < kNeighborDist * kNeighborDist)
        {
            cmX += dx;
            repX -= __fdividef(dx, lenDSqr);
            alX += d_in[iOther].z;
            cmY += dy;
            repY -= __fdividef(dy, lenDSqr);
            alY += d_in[iOther].w;
            ++numNeighbors;
        }
    }

    b.z += dt * ((kCmFactor * cmX + kAlignFactor * alX) / numNeighbors + kRepelFactor * repX);
    b.w += dt * ((kCmFactor * cmY + kAlignFactor * alY) / numNeighbors + kRepelFactor * repY);

    {
        const float lenVSqr = b.z * b.z + b.w * b.w;
        const float vScale = lenVSqr > kMaxVel * kMaxVel ? kMaxVel / sqrtf(lenVSqr) : 1.0f;
        b.z *= vScale;
        b.w *= vScale;
    }

    b.x = Rem(b.x + kVelScalar * dt * b.z, width);
    b.y = Rem(b.y + kVelScalar * dt * b.w, height);
    d_out[iBoid] = b;

    {
        const float lenVSqr = sqrtf(fmaf(b.z, b.z, fmaf(b.w, b.w, 0.000000001f)));
        b.z /= lenVSqr;
        b.w /= lenVSqr;
    }

    const float c = atan2f(b.w, b.z) * (6.0f / 6.28318530718f);
    const uchar4 rgb
    {
        static_cast<unsigned char>(255.0f * __saturatef(fabsf(Rem(c + 0.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(255.0f * __saturatef(fabsf(Rem(c + 4.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(255.0f * __saturatef(fabsf(Rem(c + 2.0f, 6.0f) - 3.0f) - 1.0f)),
        static_cast<unsigned char>(0)
    };

    int x = b.x - b.z * kBoidLen;
    const int x2 = b.x + b.z * kBoidLen;
    int y = b.y - b.w * kBoidLen;
    const int y2 = b.y + b.w * kBoidLen;

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
    CHECK_CUDA_ERRORS(cudaMalloc(reinterpret_cast<void**>(&d_boids), 2 * kNumBoids * sizeof(float4)));
    if (Reset(width, height, d_boids))
        return EXIT_FAILURE;
    CHECK_CUDA_ERRORS(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CHECK_CUDA_ERRORS(cudaFuncSetCacheConfig(BoidsKernel, cudaFuncCachePreferL1));
    CHECK_CUDA_ERRORS(cudaFuncSetAttribute(BoidsKernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
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
            if (reset && Reset(width, height, pingPong ? d_boids + kNumBoids : d_boids))
                return EXIT_FAILURE;
            CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &cgr, cs));
            size_t bufSize = 0;
            uchar4* d_pbo = nullptr;
            CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pbo), &bufSize, cgr));

            if (!traces)
                cudaMemset(d_pbo, 0, sizeof(uchar4) * width * height);

            BoidsKernel<<<kNumBlocks, kThreadsPerBlock>>>(
                d_pbo,
                pingPong ? d_boids + kNumBoids : d_boids,
                pingPong ? d_boids : d_boids + kNumBoids,
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
