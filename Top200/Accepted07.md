
[224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

[662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

[349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

[230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

[297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

[221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

[剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)

[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

[26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

[11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

[443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)

[50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)


------



[224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

```go
func calculate(s string) int {
	stack, res, num, sign := []int{}, 0, 0, 1
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			num = num*10 + int(s[i]-'0')
		} else if s[i] == '+' {
			res += sign * num
			num = 0
			sign = 1
		} else if s[i] == '-' {
			res += sign * num
			num = 0
			sign = -1
		} else if s[i] == '(' {
			stack = append(stack, res) //将前一个结果和符号压入栈中
			stack = append(stack, sign)
			res = 0 //将结果设置为0，只需在括号内计算新结果
			sign = 1
		} else if s[i] == ')' {
			res += sign * num
			num = 0
			res *= stack[len(stack)-1]
			res += stack[len(stack)-2]
			stack = stack[:len(stack)-2]
		}
	}
	if num != 0 {
		res += sign * num
	}
	return res
}

```

[662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func widthOfBinaryTree(root *TreeNode) int {
	res, left := 0, []int{1}
	var dfs func(root *TreeNode, level, pos int)
	dfs = func(root *TreeNode, level, pos int) {
		if root != nil {
			if level == len(left) {
				left = append(left, pos)
			}
			res = max(res, pos-left[level]+1) //计算每层的最大间距
			dfs(root.Left, level+1, pos*2)
			dfs(root.Right, level+1, pos*2+1)
		}
	}
	dfs(root, 0, 1)
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)



[230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

[297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

[221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

[剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)

[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

[26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

[11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

[443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)

[50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)