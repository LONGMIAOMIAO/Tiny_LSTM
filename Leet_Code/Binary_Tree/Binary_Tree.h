#pragma once
#include <memory>

namespace Tree
{
template <typename T>
struct TreeNode
{
	T val;
	std::shared_ptr<TreeNode<T>> left;
	std::shared_ptr<TreeNode<T>> right;
	TreeNode(T val) : val(val), left(nullptr), right(nullptr){};
};

} // namespace Tree