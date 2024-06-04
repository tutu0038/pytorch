#include "profiler.hpp"
#include <iostream>
#include <grpc++/grpc++.h>

#ifdef BAZEL_BUILD
#include "examples/protos/communicator.grpc.pb.h"
#else
#include "protos/communicator.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using communicator::AllgatherOutput;
using communicator::AllgatherInput;
using communicator::Allgather;

namespace c10d {


bool WorkDummy::isCompleted() {
  return true;
}

bool WorkDummy::isSuccess() const {
  return true;
}

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
ProfilerBackend::ProfilerBackend(int rank, int size)
    : Backend(rank, size) {}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> ProfilerBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {

    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    static auto invalidArgument = [](const std::string& msg) {
      TORCH_WARN(false, "ProcessGroupLLMprofiler::allgather: " + msg);
    };

    for (const auto i : c10::irange(outputTensors.size())) {
    const auto expected = inputTensors.size() * getSize();
      const auto actual = outputTensors[i].size();
    if (actual != expected) {
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) +
          " (expected length " + std::to_string(expected) + ", got " +
          std::to_string(actual) + ")");
      }
    }
   for (auto& outputTensorVec : outputTensors) {
     outputTensorVec = inputTensors;
   }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));

  GreeterClient greeter(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  std::string user("world");
  std::string reply = greeter.SayHello(user);
  TORCH_WARN(false, "Greeter received: " + reply);

  std::string Communicator(const std::string& user) {
    // Data we are sending to the server.
    AllgatherInput request;
    request.rank = rank;
    request.time = 1000;
    request.type = 0;
    
    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Allgather(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> ProfilerBackend::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  auto tensor_list = at::chunk(output_tensor, this->getSize(), 0);
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  std::vector<at::Tensor> inputs = {input_tensor};
  return this->allgather(outputs, inputs, opts);
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> ProfilerBackend::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> ProfilerBackend::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::barrier(
    const BarrierOptions& /* unused */) {

}

c10::intrusive_ptr<Work> ProfilerBackend::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> ProfilerBackend::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> ProfilerBackend::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<ProfilerBackend>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &ProfilerBackend::createBackendDummy);
}

} // namespace c10d
