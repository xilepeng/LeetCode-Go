
[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) next

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) next

[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) next

[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)


------


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

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) next


[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

### 方法一：递归

```go
func hasPathSum(root *TreeNode, sum int) bool {
	if root == nil {
		return false // 遍历到null节点
	}
	if root.Left == nil && root.Right == nil { // 遍历到叶子节点
		return sum-root.Val == 0 // 如果满足这个就返回true。否则返回false
	} // 当前递归问题 拆解成 两个子树的问题，其中一个true了就行
	return hasPathSum(root.Left, sum-root.Val) || hasPathSum(root.Right, sum-root.Val)
}
```



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



[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)




[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) next






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





[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)




[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

### 方法一：位运算 

异或运算的作用
　　参与运算的两个值，如果两个相应bit位相同，则结果为0，否则为1。

　　即：

　　0^0 = 0，

　　1^0 = 1，

　　0^1 = 1，

　　1^1 = 0

　　按位异或的3个特点：

　　（1） 0^0=0，0^1=1  0异或任何数＝任何数

　　（2） 1^0=1，1^1=0  1异或任何数－任何数取反

　　（3） 任何数异或自己＝把自己置0

```go
func singleNumber(nums []int) int {
	res := 0
	for _, num := range nums {
		res ^= num
	}
	return res
}
```