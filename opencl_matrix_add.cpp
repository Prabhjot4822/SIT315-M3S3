#define CL_TARGET_OPENCL_VERSION 300 // Specify OpenCL version (optional)
#include <CL/cl.h>                   // Include OpenCL header
#include <chrono>                    // Include for timing
#include <stdio.h>                   // Include for standard input/output
#include <stdlib.h>                  // Include for memory allocation

#define PRINT 1 // Control printing of arrays (0: disable, 1: enable)

int SZ = 100000000; // Size of the vectors

// Declare pointers for host memory (arrays)
int *v1, *v2, *v_out;

// Declare OpenCL memory objects (buffers)
cl_mem bufV1, bufV2, bufV_out;

// Declare device and OpenCL objects
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

// Event for kernel execution (optional)
cl_event event = NULL;

int err; // Variable to store error codes

// Function to create an OpenCL device
cl_device_id create_device();

// Function to set up OpenCL context, device, queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);

// Function to build the OpenCL program from source code
cl_program build_program(cl_context ctx, cl_device_id dev,
                         const char *filename);

// Function to allocate and initialize memory on the device
void setup_kernel_memory();

// Function to copy arguments to the kernel
void copy_kernel_args();

// Function to free memory allocated by the program
void free_memory();

// Function to initialize an array on the host
void init(int *&A, int size);

// Function to print an array (with size limitation for large arrays)
void print(int *A, int size);

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        SZ = atoi(argv[1]); // Get array size from command line argument (optional)
    }

    // Allocate and initialize host arrays
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ);

    // Define global work size for the kernel execution
    size_t global[1] = {(size_t)SZ};

    // Print initial arrays (optional based on PRINT flag)
    print(v1, SZ);
    print(v2, SZ);

    // Set up OpenCL environment (device, context, queue, kernel)
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl",
                                             (char *)"vector_add_ocl");

    // Allocate memory on the device for the vectors
    setup_kernel_memory();

    // Copy data from host to device memory
    copy_kernel_args();

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Enqueue kernel execution with global work size
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);

    // Wait for the kernel execution to complete
    clWaitForEvents(1, &event);

    // Copy results from device memory back to host
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0],
                        0, NULL, NULL);

    // Print the resulting array (optional based on PRINT flag)
    print(v_out, SZ);

    // Stop time measurement and calculate elapsed time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = stop - start;

    // Print kernel execution time
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count());

    // Release OpenCL resources
    free_memory();

    return 0;
}

// Function to allocate and initialize an array on the host
void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size); // Allocate memory on host

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // Assign random values between 0 and 99
    }
}

// Function to print an array (with size limitation for large arrays)
void print(int *A, int size)
{
    if (PRINT == 0)
    {
        // Early return if printing is disabled
        return;
    }

    if (PRINT == 1 && size > 15)
    {
        // Print first 5 and last 5 elements for large arrays
        for (long i = 0; i < 5; i++)
        {
            printf("%d ", A[i]); // Print element i
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++)
        {
            printf("%d ", A[i]); // Print element i
        }
    }
    else
    {
        // Print all elements for small arrays or when printing is enabled
        for (long i = 0; i < size; i++)
        {
            printf("%d ", A[i]); // Print element i
        }
    }

    // Print newline and separator line
    printf("\n----------------------------\n");
}

// Function to free memory allocated by the program (host and device)
void free_memory()
{
    // Release OpenCL memory objects (buffers)
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    // Release OpenCL objects
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    // Free host memory allocated for the arrays
    free(v1);
    free(v2);
    free(v_out);
}

// Function to copy arguments (array pointers and sizes) to the kernel
void copy_kernel_args()
{
    // Set kernel argument 0: size of the vectors (integer)
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);

    // Set kernel argument 1: buffer for vector 1 (cl_mem)
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);

    // Set kernel argument 2: buffer for vector 2 (cl_mem)
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);

    // Set kernel argument 3: buffer for the output vector (cl_mem)
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    // Check for errors during argument setting
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

// Function to allocate and initialize memory on the device for the vectors
void setup_kernel_memory()
{
    // Create OpenCL buffers (memory objects) on the device for the vectors
    // with read-write access
    bufV1 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out =
        clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    // Check for errors during buffer creation
    if ((bufV1 == NULL) || (bufV2 == NULL) || (bufV_out == NULL))
    {
        perror("Couldn't create a buffer");
        exit(1);
    }

    // Copy data from host memory (v1, v2) to device memory (buffers)
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0,
                         NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0,
                         NULL, NULL);
}

// Function to set up OpenCL device, context, queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename,
                                              char *kernelname)
{

    // Create a device (assumed the first available device)
    device_id = create_device();
    // Error handling for device creation is done in create_device()

    // Create an OpenCL context using the device
    cl_int err; // Variable to store error codes
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    // Build the OpenCL program from the source code in the specified file
    program = build_program(context, device_id, filename);

    // Create a command queue for interacting with the device
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    }

    // Create a kernel from the program using the specified kernel name
    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    }
}

// Function to build the OpenCL program from source code
cl_program build_program(cl_context ctx, cl_device_id dev,
                         const char *filename)
{

    // Read the OpenCL program source code from the specified file
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }

    // Get the size of the program source code
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    // Allocate memory to hold the program source code
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0'; // Ensure null termination

    // Read the program source code into the buffer
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Create an OpenCL program object from the source code
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer,
                                        &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }

    // Free the memory used for the program source code buffer
    free(program_buffer);

    // Build the OpenCL program for the chosen device
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        // Get the build log in case of errors
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                              &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
                              program_log, NULL);

        // Print the build log and exit on error
        printf("Program Build Error (%d):\n%s\n", err, program_log);
        free(program_log);
        exit(1);
    }

    // Return the built program object
    return program;
}

// Function to create (or retrieve) a device for OpenCL execution
cl_device_id create_device()
{

    // Get platform information
    cl_platform_id platform;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    // Get device information (prefer GPU)
    cl_device_id dev;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {

        printf("GPU not found, using CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    // . Return the chosen device handle
    return dev;
}
