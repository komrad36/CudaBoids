# CudaBoids

 See example screenshots in 'Screenshots' folder.
 
 Boids_CUDA_GL simulates bird ("boid"), fish, crowd, etc. flocking behavior,
 resulting in emergent properties. Uses CUDA for computation on the GPU
 and outputs pixels directly to an OpenGL texture with no memory transfer
 or CPU involvement for high performance. This simulation
 is primarily tuned for aesthetics, not physical accuracy, but
 careful selection of parameters can produce very flock-like emergent
 behavior. Simulation parameters can be adjusted at the top of CudaBoids.cu.

 Requires GLEW (static), GLFW (static), and CUDA (dynamic).

 Commands:
 
	Space			-	toggle traces
	P			-	pause
	R			-	reset boids
	T			-	toggle attraction/repulsion
	ESC			-	quit
	1 mouse button		-	weak attraction/repulsion
	2 mouse buttons		-	strong attracton/repulsion
