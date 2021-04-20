
[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)


------



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



[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)
### 方法一：递归
```go
func isSymmetric(root *TreeNode) bool {
	return check(root, root)
}
func check(p, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	return p.Val == q.Val && check(p.Left, q.Right) && check(p.Right, q.Left)
}
```
### 方法二：迭代
```go
func isSymmetric(root *TreeNode) bool {
	q := []*TreeNode{root, root}
	for 0 < len(q) {
		l, r := q[0], q[1]
		q = q[2:]
		if l == nil && r == nil {
			continue
		}
		if l == nil || r == nil {
			return false
		}
		if l.Val != r.Val {
			return false
		}
		q = append(q, l.Left)
		q = append(q, r.Right)

		q = append(q, l.Right)
		q = append(q, r.Left)
	}
	return true
}
```




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