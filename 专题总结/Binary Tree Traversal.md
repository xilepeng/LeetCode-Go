

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)


------


![Binary Tree Traversal Iteration Implementation.png](http://ww1.sinaimg.cn/large/007daNw2ly1gqnx5to70yj30pd2bie81.jpg)




[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func preorderTraversal(root *TreeNode) []int {
	res, stack := []int{}, []*TreeNode{}
	p := root
	for len(stack) != 0 || p != nil {
		if p != nil {
			stack = append(stack, p)
			res = append(res, p.Val)
			p = p.Left
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			p = node.Right
		}
	}
	return res
}
```




[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func inorderTraversal(root *TreeNode) []int {
	res, stack := []int{}, []*TreeNode{}
	p := root
	for len(stack) > 0 || p != nil {
		if p != nil {
			stack = append(stack, p)
			p = p.Left
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res = append(res, node.Val)
			p = node.Right
		}
	}
	return res
}
```



[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func postorderTraversal(root *TreeNode) []int {
	res, stack := []int{}, []*TreeNode{}
	p := root
	for len(stack) > 0 || p != nil {
		if p != nil {
			stack = append(stack, p)
			res = append(append([]int{}, p.Val), res...) //反转先序遍历
			p = p.Right                                  //反转先序遍历
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			p = node.Left //反转先序遍历
		}
	}
	return res
}
```



[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

``` go

```







## 其他解法

``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	stack, res := []*TreeNode{root}, []int{}
	for len(stack) != 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, curr.Val)
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
	}
	return res
}
```


``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func postorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	stack, res := []*TreeNode{root}, []int{}
	for len(stack) != 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(append([]int{}, curr.Val), res...)
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}

	}
	return res
}
```