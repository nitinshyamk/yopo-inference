#pragma once
#include "utilities.h"

double calculate_torch_accuracy(torch::Tensor output, torch::Tensor target)
{
	if (output.dim() != 2 || target.dim() != 2 || output.size(0) != target.size(0))
		throw std::invalid_argument("Incompatible label dimensions");
	auto batch_size = output.size(0);

	torch::Tensor predictions;
	std::tie(std::ignore, predictions) = output.topk(
		1 /* get top logit value */,
		1 /* by going along the dimension of logits*/,
		true /* and take max */,
		true /* leave default */);
	assert_equal_content_count(target, predictions);
	auto target_reshaped = target.view({ 1, -1 }).expand_as(predictions);
	auto is_correct = predictions.eq_(target_reshaped);
	auto ans = is_correct.view(-1).sum(0);
	auto val = ans.mul_(100.0 / batch_size);
	if (get_element_count(val) != 1)
		throw std::runtime_error("Batch accuracy not correct");
	return val[0].item<double>();
}

void assert_equal_content_count(torch::Tensor a, torch::Tensor b)
{
	if ((a.dim() < 1 || b.dim() < 1 && a.dim() != b.dim()) || (get_element_count(a) != get_element_count(b)))
		throw std::invalid_argument("Element counts are unequal");
}

int get_element_count(torch::Tensor a)
{
	if (a.dim() < 1) return 0;
	auto count = a.size(0);
	for (int d = 1; d < a.dim(); ++d)
		count *= a.size(d);
	return count;
}