#pragma once
#include <torch/torch.h>

namespace nn = torch::nn;

struct StackSequentialImpl : torch::nn::SequentialImpl {
	using SequentialImpl::SequentialImpl;

	torch::Tensor forward(torch::Tensor x) {
		return SequentialImpl::forward(x);
	}
};
TORCH_MODULE(StackSequential);

struct SmallCNNImpl : nn::Module
{
public:
	SmallCNNImpl(double drop_rate = 0.5, size_t numlabels = 10) : _numlabels(numlabels)
	{
		_l1 = StackSequential(
			create_conv2d(this->_numchannels, 32, 3),
			nn::ReLU());

		_feature_extractor = StackSequential(
			create_conv2d(32, 32, 3),
			nn::ReLU(),
			nn::MaxPool2d(nn::MaxPool2dOptions({ 2, 2 })),
			create_conv2d(32, 64, 3),
			nn::ReLU(),
			create_conv2d(64, 64, 3),
			nn::ReLU(),
			nn::MaxPool2d(nn::MaxPool2dOptions({ 2, 2 }))
		);

		auto lin3 = nn::Linear(200, _numlabels);
		_classifier = StackSequential(
			nn::Linear(64 * 4 * 4, 200),
			nn::ReLU(),
			nn::Dropout(drop_rate),
			nn::Linear(200, 200),
			nn::ReLU(),
			lin3);

		register_module("_l1", _l1);
		register_module("_feature_extractor", _feature_extractor);
		register_module("_classifier", _classifier);

		auto lin3params = lin3->named_parameters();
		nn::init::constant_(lin3params["weight"], 0);
		nn::init::constant_(lin3params["bias"], 0);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		if (x.dim() != 4 || x.size(1) != 1 || x.size(2) != 28 || x.size(3) != 28)
			throw std::invalid_argument("Incorrectly sized input tensor. Should have dimensions BatchSize X Channel (1) X Height (28) X Width (28)");
		auto y = _l1->forward(x);
		auto l1out = y; l1out.requires_grad_(); l1out.retain_grad();
		auto features = this->_feature_extractor->forward(y);
		auto logits = this->_classifier->forward(features.view({ -1, 64 * 4 * 4 }));
		return logits;
	}

private:
	// data
	size_t _numchannels = 1;
	size_t _numlabels = 10;

	// layers
	StackSequential _l1{ nullptr };
	StackSequential _feature_extractor{ nullptr };
	StackSequential _classifier{ nullptr };

	nn::Conv2d create_conv2d(size_t inchannel, size_t outchannel, size_t kernel)
	{
		if (inchannel < 1 || outchannel < 1 || kernel < 1) throw std::invalid_argument("invalid specifications");
		auto layer = nn::Conv2d(inchannel, outchannel, kernel);

		auto params = layer->named_parameters();
		if (params.contains("weight")) nn::init::kaiming_normal_(params["weight"]);
		else throw std::invalid_argument("unable to locate module's weight matrix");
		if (params.contains("bias")) nn::init::constant_(params["bias"], 0);
		return layer;
	}
};

TORCH_MODULE(SmallCNN);
