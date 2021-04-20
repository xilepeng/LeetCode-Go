
[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)


### 方法一：动态规划

![截屏2021-04-20 10.00.31.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpq0altspuj31hw0t4dk1.jpg)

![截屏2021-04-20 10.01.56.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpq0assvhfj31sc0scqb0.jpg)


```go
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}
	for j := 0; j < n; j++ {
		dp[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}
```

复杂度分析

- 时间复杂度：O(mn)。

- 空间复杂度：O(mn)，即为存储所有状态需要的空间。注意到 dp[i][j] 仅与第 i 行和第 i−1 行的状态有关，因此我们可以使用滚动数组代替代码中的二维数组，使空间复杂度降低为 O(n)。此外，由于我们交换行列的值并不会对答案产生影响，因此我们总可以通过交换 m 和 n 使得 m≤n，这样空间复杂度降低至 O(min(m,n))。

### 优化

```go
func uniquePaths(m int, n int) int {
	dp := make([]int, n)
	for j := 0; j < n; j++ {
		dp[j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}
```
复杂度分析

- 时间复杂度：O(mn)。

- 空间复杂度：O(n)，即为存储所有状态需要的空间。注意到 dp[i][j] 仅与第 i 行和第 i−1 行的状态有关，因此我们可以使用滚动数组代替代码中的二维数组，使空间复杂度降低为 O(n)。此外，由于我们交换行列的值并不会对答案产生影响，因此我们总可以通过交换 m 和 n 使得 m≤n，这样空间复杂度降低至 O(min(m,n))。






[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)


[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthLargest(root *TreeNode, k int) (res int) {
    var dfs func(*TreeNode)

    dfs = func(root *TreeNode) {
        if root == nil {
            return
        }
        dfs(root.Right)
        k--
        if k == 0 { res = root.Val}
        dfs(root.Left) 
    }
    
    dfs(root)
    return res
}
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthLargest(root *TreeNode, k int) int {
    res := []int{}
    var dfs func(*TreeNode)

    dfs = func(root *TreeNode) {
        if root == nil {
            return
        }
        dfs(root.Right)
        res = append(res, root.Val)
        dfs(root.Left)
    }

    dfs(root)
    return res[k-1]
}
```


[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

### 方法一：用翻转代替旋转

![截屏2021-04-20 17.50.27.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpqcvh0n5pj314w0mkadk.jpg)

![截屏2021-04-20 17.50.41.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpqcvp2lc4j318e0o60xd.jpg)

```go
func rotate(matrix [][]int) {
	n := len(matrix)
	// 水平翻转
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	// 主对角线翻转
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}
```


[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)


[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)



[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)


[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

![截屏2021-04-19 12.05.24.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpox9o0ps6j30wg0di75x.jpg)

```go
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	dummy := &ListNode{0, head}
	cur := dummy
	for cur.Next != nil && cur.Next.Next != nil {
		if cur.Next.Val == cur.Next.Next.Val {
			x := cur.Next.Val
			for cur.Next != nil && cur.Next.Val == x {
				cur.Next = cur.Next.Next
			}
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}
```


[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)