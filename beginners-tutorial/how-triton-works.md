## Questions I was curious about

### 1. What is a "Triton Instance"?

```cpp
// triton-inferene-server/core/src/backend_model_instance.h
class TritonModelInstance {
 public:
  struct SecondaryDevice {
    SecondaryDevice(const std::string kind, const int64_t id)
        : kind_(kind), id_(id)
    {
    }
    const std::string kind_;
    const int64_t id_;
  };
```

The `TritonModelInstance` class is likely responsible for representing an instance of a model within the Triton Inference Server. Each instance of a model can be deployed on a specific hardware device (e.g., GPU, CPU) and is capable of processing inference requests independently.

The `SecondaryDevice` struct is nested within the `TritonModelInstance` class and is used to represent additional hardware resources that a model instance might use. Here's a breakdown of its components:

**Constructor**

```
SecondaryDevice(const std::string kind, const int64_t id)
```

- Initializes a `SecondaryDevice` object with a specific kind (type) and id (identifier).
- `kind_`: A string representing the type of device (e.g., "GPU", "CPU").
- `id_`: An integer representing the unique identifier of the device.

By encapsulating device information within the `SecondaryDevice` struct, Triton can manage and track the resources used by each model instance more effectively. This helps in optimizing resource allocation and ensuring that the server can handle multiple instances efficiently.

### 2. What happens in the code when you make a instance?

`triton-inferene-server/core/src/backend_model_instance.cc`
1. **`TritonModelInstance::TritonModelInstance`**

```cpp
//constructor
TritonModelInstance::TritonModelInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const triton::common::HostPolicyCmdlineConfig& host_policy,
    const TritonServerMessage& host_policy_message,
    const std::vector<SecondaryDevice>& secondary_devices)
    : model_(model), name_(name), signature_(signature), kind_(kind),
      device_id_(device_id), host_policy_(host_policy),
      host_policy_message_(host_policy_message), profile_names_(profile_names),
      passive_(passive), secondary_devices_(secondary_devices), state_(nullptr)
{
#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    // Use an ID in the metric only for GPU instances. Otherwise use
    // METRIC_REPORTER_ID_CPU to indicate no device should be reported in the
    // metric.
    const int id = (kind_ == TRITONSERVER_INSTANCEGROUPKIND_GPU)
                       ? device_id_
                       : METRIC_REPORTER_ID_CPU;
    // Let every metric reporter know if caching is enabled to correctly include
    // cache miss time into request duration on cache misses.
    const bool response_cache_enabled =
        model_->ResponseCacheEnabled() &&
        model_->Server()->ResponseCacheEnabled();
    MetricModelReporter::Create(
        model_->ModelId(), model_->Version(), id, response_cache_enabled,
        model_->Config().metric_tags(), &reporter_);
  }
#endif  // TRITON_ENABLE_METRICS
}

// destructor
TritonModelInstance::~TritonModelInstance()
{
  //@Soeun : If a backend thread exists, stop the backend thread
  if (triton_backend_thread_.get() != nullptr) {
    triton_backend_thread_->StopBackendThread();
  }

  model_->Server()->GetRateLimiter()->UnregisterModelInstance(this);

  // Model finalization is optional...
  if (model_->Backend()->ModelInstanceFiniFn() != nullptr) {
    LOG_TRITONSERVER_ERROR(
        model_->Backend()->ModelInstanceFiniFn()(
            reinterpret_cast<TRITONBACKEND_ModelInstance*>(this)),
        "failed finalizing model instance");
  }
}

```

 **Parameters:**
- **`TritonModel* model`**: A pointer to the parent model object.
- **`const std::string& name`**: The name of the model instance.
- **`const Signature& signature`**: The signature of the model instance, which likely contains information about the model's inputs and outputs.
- **`const TRITONSERVER_InstanceGroupKind kind`**: The kind of instance group (e.g., GPU or CPU).
- **`const int32_t device_id`**: The ID of the device (GPU or CPU) to which this instance is assigned.
- **`const std::vector<std::string>& profile_names`**: A list of profile names associated with the model instance.
- **`const bool passive`**: A flag indicating whether this instance is passive.
- **`const triton::common::HostPolicyCmdlineConfig& host_policy`**: Host policy configurations.
- **`const TritonServerMessage& host_policy_message`**: Host policy messages.
- **`const std::vector<SecondaryDevice>& secondary_devices`**: A list of secondary devices associated with this instance.


2. **`TritonModelInstance::CreateInstance`**

```cpp
// triton-inferene-server/core/src/backend_model_instance.cc

Status
TritonModelInstance::CreateInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    TRITONSERVER_InstanceGroupKind kind, int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const std::string& host_policy_name,
    const inference::ModelRateLimiter& rate_limiter_config,
    const std::vector<SecondaryDevice>& secondary_devices,
    std::shared_ptr<TritonModelInstance>* triton_model_instance)
{
  static triton::common::HostPolicyCmdlineConfig empty_host_policy;
  const triton::common::HostPolicyCmdlineConfig* host_policy =
      &empty_host_policy;
  //@Soeun : Retrieves the host policy configuration 
  const auto policy_it = model->HostPolicyMap().find(host_policy_name);
  if (policy_it != model->HostPolicyMap().end()) {
    host_policy = &policy_it->second;
  }

  //@Soeun : Sets the NUMA(Non-Uniform Memory Access) configuration on the thread based on the host policy
  RETURN_IF_ERROR(SetNumaConfigOnThread(*host_policy));
  //@Soeun : Calls `ConstructAndInitializeInstance` to create and initialize the model instance.
  auto err = ConstructAndInitializeInstance(
      model, name, signature, kind, device_id, profile_names, passive,
      host_policy_name, *host_policy, rate_limiter_config, secondary_devices,
      triton_model_instance);
  //@Soeun : Resets the NUMA memory policy after instance construction.
  RETURN_IF_ERROR(ResetNumaMemoryPolicy());
  RETURN_IF_ERROR(err);

  // When deploying on GPU, we want to make sure the GPU memory usage
  // is within allowed range, otherwise, stop the creation to ensure
  // there is sufficient GPU memory for other use.
  // We check the usage after loading the instance to better enforcing
  // the limit. If we check before loading, we may create instance
  // that occupies the rest of available memory which against the purpose
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    RETURN_IF_ERROR(VerifyModelLoadGpuFraction(
        name, kind, device_id, model->BackendConfigMap()));
  }

  return Status::Success;
}
```

3. **`TritonModelInstance::ConstructAndInitializeInstance`**

The `ConstructAndInitializeInstance` method performs the actual creation and initialization of a `TritonModelInstance`. This includes setting up the model instance, initializing it with the backend, and registering it with the rate limiter.

```cpp
Status
TritonModelInstance::ConstructAndInitializeInstance(
    TritonModel* model, const std::string& name, const Signature& signature,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const std::vector<std::string>& profile_names, const bool passive,
    const std::string& host_policy_name,
    const triton::common::HostPolicyCmdlineConfig& host_policy,
    const inference::ModelRateLimiter& rate_limiter_config,
    const std::vector<SecondaryDevice>& secondary_devices,
    std::shared_ptr<TritonModelInstance>* triton_model_instance)
{
  // Create the JSON representation of the backend configuration.
  triton::common::TritonJson::Value host_policy_json(
      triton::common::TritonJson::ValueType::OBJECT);
  triton::common::TritonJson::Value policy_setting_json(
      host_policy_json, triton::common::TritonJson::ValueType::OBJECT);
  for (const auto& pr : host_policy) {
    RETURN_IF_ERROR(policy_setting_json.AddString(pr.first.c_str(), pr.second));
  }

  RETURN_IF_ERROR(host_policy_json.Add(
      host_policy_name.c_str(), std::move(policy_setting_json)));
  TritonServerMessage host_policy_message(host_policy_json);

// @Soeun : Creates a new `TritonModelInstance` object with the provided parameters.
  std::unique_ptr<TritonModelInstance> local_instance(new TritonModelInstance(
      model, name, signature, kind, device_id, profile_names, passive,
      host_policy, host_policy_message, secondary_devices));

// @Soeun : If a backend initialization function is provided, it is called to initialize the model instance.
  TRITONBACKEND_ModelInstance* triton_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(local_instance.get());

  // Instance initialization is optional... We must set set shared
  // library path to point to the backend directory in case the
  // backend library attempts to load additional shared libraries.
  if (model->Backend()->ModelInstanceInitFn() != nullptr) {
    // We must set set shared library path to point to the backend directory in
    // case the backend library attempts to load additional shared libraries.
    // Currently, the set and reset function is effective only on Windows, so
    // there is no need to set path on non-Windows.
    // However, parallel model loading will not see any speedup on Windows and
    // the global lock inside the SharedLibrary is a WAR.
    // [FIXME] Reduce lock WAR on SharedLibrary (DLIS-4300)
#ifdef _WIN32
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->SetLibraryDirectory(model->Backend()->Directory()));
#endif

    TRITONSERVER_Error* err =
        model->Backend()->ModelInstanceInitFn()(triton_instance);

#ifdef _WIN32
    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
#endif
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  if (!passive) {
    RETURN_IF_ERROR(local_instance->GenerateWarmupData());
    RETURN_IF_ERROR(model->Server()->GetRateLimiter()->RegisterModelInstance(
        local_instance.get(), rate_limiter_config));
    RETURN_IF_ERROR(local_instance->SetBackendThread(
        kind, device_id, model->DeviceBlocking()));
  }

  triton_model_instance->reset(local_instance.release());

  return Status::Success;
}
```

4.  **`TritonModelInstance::SetBackendThread`**
- Set up and manage the backend thread for the model instance.

```cpp
Status
TritonModelInstance::SetBackendThread(
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
    const bool device_blocking)
{
  if (ShareBackendThread(device_blocking, kind)) {
    auto device_instances = model_->GetInstancesByDevice(device_id);
    if (!device_instances.empty()) {
      LOG_VERBOSE(1) << "Using already started backend thread for " << Name()
                     << " on device " << device_id;
      triton_backend_thread_ = device_instances[0]->triton_backend_thread_;
    }
  }
  if (triton_backend_thread_.get() == nullptr) {
    std::unique_ptr<TritonBackendThread> local_backend_thread;
    RETURN_IF_ERROR(TritonBackendThread::CreateBackendThread(
        Name(), this, 0 /* nice */, device_id, &local_backend_thread));
    triton_backend_thread_ = std::move(local_backend_thread);
  } else {
    triton_backend_thread_->AddModelInstance(this);
  }
  RETURN_IF_ERROR(triton_backend_thread_->InitAndWarmUpModelInstance(this));

  return Status::Success;
}
```


6. **`TritonModelInstance::Schedule`**

```cpp
Status
TritonModelInstance::Schedule(
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  // Prepare requests for execution, respond to requests if any error occur.
  RETURN_IF_ERROR(PrepareRequestsOrRespond(requests));

  // Use a thread local vector to avoid needing to malloc each
  // time an inference is run.
  thread_local std::vector<TRITONBACKEND_Request*> triton_requests(1024);
  triton_requests.clear();
  for (auto& r : requests) {
    triton_requests.push_back(
        reinterpret_cast<TRITONBACKEND_Request*>(r.release()));
  }

  Execute(triton_requests);
  return Status::Success;
}
```
