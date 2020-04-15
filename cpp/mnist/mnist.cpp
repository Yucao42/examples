#include <torch/torch.h>
#include <cuda.h>
#include <stdio.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "papi.h"
#include "papi_test.h"

#define PAPI 1

#define NUM_EVENTS 2
// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net : torch::nn::Module {
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

template <typename DataLoader>
void train(
    int32_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     at::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}


int main(int argc, char** argv){
  #ifdef PAPI
  	int quiet=0;
  	int retval, i;
  	int EventSet = PAPI_NULL;
  	long long values[NUM_EVENTS];
  	/* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT 
  	   FOR THE CUDA DEVICE YOU ARE RUNNING ON.
  	   RUN papi_native_avail to get a list of CUDA events that are 
  	   supported on your machine */
          //char *EventName[] = { "PAPI_FP_OPS" };
          char const *EventName[] = { "cuda:::event:elapsed_cycles_sm:device=0", "cuda:::event:active_warps:device=0" };
  	int events[NUM_EVENTS];
  	int eventCount = 0;
  
  
  	/* Set TESTS_QUIET variable */
  	quiet=tests_quiet( argc, argv );
  	
  	/* PAPI Initialization */
  	retval = PAPI_library_init( PAPI_VER_CURRENT );
  	if( retval != PAPI_VER_CURRENT ) {
  		if (!quiet) printf("PAPI init failed\n");
  		test_fail(__FILE__,__LINE__,
  			"PAPI_library_init failed", 0 );
  	}
  
  	if (!quiet) {
  		printf( "PAPI_VERSION     : %4d %6d %7d\n",
  			PAPI_VERSION_MAJOR( PAPI_VERSION ),
  			PAPI_VERSION_MINOR( PAPI_VERSION ),
  			PAPI_VERSION_REVISION( PAPI_VERSION ) );
  	}
  
  	/* convert PAPI native events to PAPI code */
  	for( i = 0; i < NUM_EVENTS; i++ ){
                  retval = PAPI_event_name_to_code( (char *)EventName[i], &events[i] );
  		printf( "PAPI RETURNS     : %4d\n", retval);
  		if( retval != PAPI_OK ) {
  			fprintf( stderr, "PAPI_event_name_to_code failed\n" );
  			continue;
  		}
  		eventCount++;
  		if (!quiet) printf( "Name %s --- Code: %#x\n", EventName[i], events[i] );
  	}
  
  	/* if we did not find any valid events, just report test failed. */
  	if (eventCount == 0) {
  		if (!quiet) printf( "Test FAILED: no valid events found.\n");
  		test_skip(__FILE__,__LINE__,"No events found",0);
  		return 1;
  	}
  	
  	retval = PAPI_create_eventset( &EventSet );
  	if( retval != PAPI_OK ) {
  		if (!quiet) printf( "PAPI_create_eventset failed\n" );
  		test_fail(__FILE__,__LINE__,"Cannot create eventset",retval);
  	}	
  
          // If multiple GPUs/contexts were being used, 
          // you need to switch to each device before adding its events
          // e.g. cudaSetDevice( 0 );
  	retval = PAPI_add_events( EventSet, events, eventCount );
  	if( retval != PAPI_OK ) {
  		fprintf( stderr, "PAPI_add_events failed\n" );
  	}
  
  	retval = PAPI_start( EventSet );
  	if( retval != PAPI_OK ) {
  		fprintf( stderr, "PAPI_start failed\n" );
  	}
  #endif

  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    start = std::chrono::steady_clock::now();
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    end = std::chrono::steady_clock::now();
    std::cout << "\n[TIME] Training time in milleseconds : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;
    start = end;
    test(model, device, *test_loader, test_dataset_size);
    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Testing time in milleseconds : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;
  }
  #ifdef PAPI
  	retval = PAPI_stop( EventSet, values );
  	if( retval != PAPI_OK )
  		fprintf( stderr, "PAPI_stop failed\n" );
  
  	retval = PAPI_cleanup_eventset(EventSet);
  	if( retval != PAPI_OK )
  		fprintf(stderr, "PAPI_cleanup_eventset failed\n");
  
  	retval = PAPI_destroy_eventset(&EventSet);
  	if (retval != PAPI_OK)
  		fprintf(stderr, "PAPI_destroy_eventset failed\n");
  
  	PAPI_shutdown();
  
  	for( i = 0; i < eventCount; i++ )
  		if (!quiet) printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
  #endif
}
