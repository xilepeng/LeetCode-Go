
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

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

### 思路
1. 有序数组分成了左右2个小的有序数组，而实际上要找的是右边有序数组的最小值
2. 如果中间值大于右边的最大值，说明中间值还在左边的小数组里，需要left向右移动
3. 否则就是中间值小于等于当前右边最大值，mid 已经在右边的小数组里了，但是至少说明了当前右边的right值不是最小值了或者不是唯一的最小值，需要慢慢向左移动一位。


```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right--
		}
	}
	return nums[left]
}
```

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] < nums[right] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return nums[left]
}
```



[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)


[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)


[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)




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