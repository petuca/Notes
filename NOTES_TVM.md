# LLVM with TVM Notes

Apache TVM is a deep learning compiler that enables access to high-performance machine learning anywhere for everyone. TVM empowers users to leverage community-driven ML-based optimizations to push the limits and amplify the reach of their research and development, which in turn raises the collective performance of all ML, while driving its costs down.

Codebase Structure and explanation of connection between LLVM and TVM
------------------

The code is composed of several important parts:
src - C++ code for operator compilation and deployment runtimes
src/relay - Implementation of Relay, a new functional IR for deep learning framework
python - Python frontend that wraps C++ functions and objects implemented in src
src/topi - Compute definitions and backend schedules for standard neural network operators.

src/relay is the component that manages a computational graph, and nodes in a graph are compiled and executed using infrastructure implemented in the rest of src. python provides python bindings for the C++ API and driver code that users can use to execute compilation. Operators corresponding to each node are registered in src/relay/op. Implementations of operators are in topi, and they are coded in either C++ or Python.

When user write some code and then want to compile that code, he calls the relay.build(...) function and for each node in the graph looks up an operator implementation by querying the operator registry, generate a compute expression and a schedule for the operator, compile the operator into object code.

All the stuff of operating are implemented in C++, Python is used for the user side. 

tvm.build(), defined in python/tvm/driver/build_module.py, takes a schedule, input and output Tensor, and a target, and returns a tvm.runtime.Module object. A tvm.runtime.Module object contains a compiled function which can be invoked with function call syntax. Target is specified as "llvm". Also can be specified as "CUDA". (OpenCL, Vulkan, ROCm)
The process of tvm.build() can be divided into two steps:
Lowering, where a high level, initial loop nest structures are transformed into a final, low level IR
Code generation, where target machine code is generated from the low level IR

Code generation is done by build_module() function, defined in python/tvm/target/codegen.py. On the C++ side, code generation is implemented in src/target/codegen subdirectory. build_module() Python function will reach Build() function below in src/target/codegen/codegen.cc

If you choose target as backend that uses LLVM, which includes x86, ARM, NVPTX and AMDGPU, code generation is done primarily by CodeGenLLVM class defined in src/codegen/llvm/codegen_llvm.cc. CodeGenLLVM translates TVM IR into LLVM IR, runs a number of LLVM optimization passes, and generates target machine code.

You can see [codegen_llvm.cc](https://github.com/apache/tvm/blob/main/src/target/llvm/codegen_llvm.cc)

The Build() function in src/codegen/codegen.cc returns a runtime::Module object, defined in include/tvm/runtime/module.h and src/runtime/module.cc. A Module object is a container for the underlying target specific ModuleNode object. Each backend implements a subclass of ModuleNode to add target specific runtime API calls.

You can see [module.cc](https://github.com/apache/tvm/blob/main/src/runtime/module.cc)

The LLVM backend implements LLVMModuleNode in src/codegen/llvm/llvm_module.cc, which handles JIT execution of compiled code.

You can see [llvm_module.cc](https://github.com/apache/tvm/blob/main/src/target/llvm/llvm_module.cc)

The target module contains all the code generators that translate an IRModule to a target runtime.Module. It also provides a common Target class that describes the target.

The target translation phase transforms an IRModule to the corresponding target executable format. For backends such as x86 and ARM, we use the LLVM IRBuilder to build in-memory LLVM IR. Finally, we support direct translations of a Relay function (sub-graph) to specific targets via external code generators. It is important that the final code generation phase is as lightweight as possible. Vast majority of transformations and lowering should be performed before the target translation phase.
![Screenshot from 2022-07-07 16-43-48](https://user-images.githubusercontent.com/104573172/177802126-e3e66a17-b778-43ef-a8c3-a02bfd9b72d8.png)

![Screenshot from 2022-07-07 16-45-29](https://user-images.githubusercontent.com/104573172/177802450-7e6f845f-bc7b-408a-adc7-3042b83a90d9.png)

The target code generators construct a Module consisting of one or more PackedFunc, from an IRModule.
The Target object is a lookup table of properties about a physical device, its hardware/driver limits, and its capabilities. The Target is accessible both during optimization and code generation stages. While the same Target class is used for all runtime targets, each runtime target may need to add target-specific options.
The code generators take an optimized IRModule and converts it into an executable representation. Each code generator must be registered in order to be used by the TVM framework. This is done by registering a function named "target.build.foo", where foo is the same name as was used in the TVM_REGISTER_TARGET_KIND definition above.
The code generator takes two arguments. The first is the IRModule to compile, and the second is the Target that describes the device on which the code should run. Because the environment performing the compilation is not necessarily the same as the environment that will be executing the code, code generators should not perform any attribute lookups on the device itself, and should instead access parameters stored in the Target.


User Side
------------------
Compiling model in command line:
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx

The chosen target is llvm.

Tune the model in command line:
tvmc tune \
--target "llvm -mcpu=broadwell" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx

The chosen target is llvm.

[Here](https://tvm.apache.org/docs/reference/api/python/target.html) you can see details about tvm.target.Target class for choosing the right target.

Compiling the model in Python:
input_name = "data"
shape_dict = {input_name: img_data.shape}
target="llvm"
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
    
This segment of code calling the right code for target machine code

Additional: When building the TVM on your device, you can specify what target you want to install llvm/cuda/opencl/rocm/vulkan

When we are done with building the module we got an target machine code ready for use and runing.

IR Module
-------------
![Screenshot from 2022-07-08 14-49-06](https://user-images.githubusercontent.com/104573172/177995233-925f8b2d-8947-4c05-a361-b4ddc5f08335.png)


An IRModule is the central data structure in TVM, which contains deep learning programs. It is the basic object of interest of IR transformation and model building.


After building the module:

mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

Output:
<class 'tvm.driver.build_module.OperatorModule'>



