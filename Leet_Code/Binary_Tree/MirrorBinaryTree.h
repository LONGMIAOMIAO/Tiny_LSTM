#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "Binary_Tree.h"
namespace Tree
{
namespace MirrorBinaryTree
{
//  way1 middle traversal
//  This Way is not good, because the
template <typename T>
void middleTraversal(std::shared_ptr<TreeNode<T>> node, std::vector<T> &vec)
{
    if (node == nullptr)
    {
        return;
    }

    middleTraversal(node->left, vec);
    vec.push_back(node->val);
    middleTraversal(node->right, vec);
}
template <typename T>
bool isMirror(std::shared_ptr<TreeNode<T>> node)
{
    std::vector<T> vec;
    middleTraversal<T>(node, vec);
    std::for_each(vec.begin(), vec.end(), [](T s) { std::cout << s << std::endl; });
}

//  way2 recursion
template <typename T>
bool isMirror_In( std::shared_ptr<TreeNode<T>> left, std::shared_ptr<TreeNode<T>> right)
{
    if ( left == nullptr && right == nullptr )
    {
        return true;
    }
    else if ( left == nullptr || right == nullptr )
    {
        return false;
    }
    
    return  (left->val == right->val) && isMirror_In( left->left, right->right ) && isMirror_In( left->right, right->left );
}

template <typename T>
bool isMirror_2(std::shared_ptr<TreeNode<T>> node )
{
    if (node == nullptr)
    {
        return true;
    }
    return isMirror_In( node->left, node->right );
}

void Test()
{
    auto n_1 = std::make_shared<TreeNode<int>>(1);
    auto n_2 = std::make_shared<TreeNode<int>>(2);
    auto n_3 = std::make_shared<TreeNode<int>>(2);
    auto n_4 = std::make_shared<TreeNode<int>>(4);
    auto n_5 = std::make_shared<TreeNode<int>>(4);

    n_1->left = n_2;
    n_1->right = n_3;
    n_2->left = n_4;
    n_3->right = n_5;

    //  way one
    //  isMirror<int>(n_1);

    //  way two
    std::cout << isMirror_2<int>(n_1) << std::endl;
}
} // namespace MirrorBinaryTree
} // namespace Tree