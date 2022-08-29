
[242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

[191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)

[补充题9. 36进制加法](https://mp.weixin.qq.com/s/bgD1Q5lc92mX7RNS1L65qA)

[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

[剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

[剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

[91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

[1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

[剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

[补充题14. 阿拉伯数字转中文数字]()

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

[61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

[86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

[208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

[114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

[556. 下一个更大元素 III](https://leetcode-cn.com/problems/next-greater-element-iii/)



------


[242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

[191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)


[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/) 扩展

``` go
func addStrings(num1 string, num2 string) string {
	carry := 0
	res := ""
	i, j := len(num1)-1, len(num2)-1
	for ; i >= 0 || j >= 0 || carry != 0; i, j = i-1, j-1 {
		var x, y int
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		sum := x + y + carry
		res = strconv.Itoa(sum%10) + res
		carry = sum / 10
	}
	return res
}
```

[补充题9. 36进制加法](https://mp.weixin.qq.com/s/bgD1Q5lc92mX7RNS1L65qA)

``` go
func addStrings(num1 string, num2 string) string {
	carry := 0
	res := ""
	i, j := len(num1)-1, len(num2)-1
	for ; i >= 0 || j >= 0 || carry != 0; i, j = i-1, j-1 {
		var x, y int
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		sum := x + y + carry
		res = strconv.Itoa(sum%10) + res
		carry = sum / 10
	}
	return res
}
```


[152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

[剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

[剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

[91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

[1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

[剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

[补充题14. 阿拉伯数字转中文数字]()

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

[61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

[86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

[208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

[114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

[556. 下一个更大元素 III](https://leetcode-cn.com/problems/next-greater-element-iii/)



