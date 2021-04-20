[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)


[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)


[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)


[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)


[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)


[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)


[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)


[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)


[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)


[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)


------

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