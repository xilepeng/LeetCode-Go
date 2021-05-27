

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

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

```go
func exist(board [][]byte, word string) bool {
	var dfs func(int, int, int) bool

	dfs = func(y, x, i int) bool {
		if i == len(word) {
			return true
		}
		if y < 0 || x < 0 || y == len(board) || x == len(board[y]) {
			return false
		}
		if board[y][x] != word[i] {
			return false
		}
		board[y][x] ^= 255
		exist := dfs(y, x+1, i+1) || dfs(y, x-1, i+1) || dfs(y+1, x, i+1) || dfs(y-1, x, i+1)
		board[y][x] ^= 255
		return exist
	}

	for y := 0; y < len(board); y++ {
		for x := 0; x < len(board[y]); x++ {
			if dfs(y, x, 0) {
				return true
			}
		}
	}
	return false
}
```

```go
func exist(board [][]byte, word string) bool {
	for y := 0; y < len(board); y++ {
		for x := 0; x < len(board[y]); x++ {
			if dfs(board, y, x, word, 0) {
				return true
			}
		}
	}
	return false
}
func dfs(board [][]byte, y int, x int, word string, i int) bool {
	if i == len(word) {
		return true
	}
	if y < 0 || x < 0 || y == len(board) || x == len(board[y]) {
		return false
	}
	if board[y][x] != word[i] {
		return false
	}
	board[y][x] ^= 255
	exist := dfs(board, y, x+1, word, i+1) || dfs(board, y, x-1, word, i+1) ||
		dfs(board, y+1, x, word, i+1) || dfs(board, y-1, x, word, i+1)
	board[y][x] ^= 255
	return exist
}
```

[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)



```go
func isPalindrome(x int) bool {
	if x < 0 || (x%10 == 0 && x != 0) { //第1位不是0，最后一位是0
		return false
	}
	rev := 0
	for x > rev {
		rev = rev*10 + x%10
		x /= 10
	}
	return x == rev || x == rev/10 //奇数去除处于中位的数字
}

```

复杂度分析

时间复杂度：O(logn)，对于每次迭代，我们会将输入除以 10，因此时间复杂度为 O(logn)。
空间复杂度：O(1)。我们只需要常数空间存放若干变量。



```go
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	s := strconv.Itoa(x)
	left, right := 0, len(s)-1
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
```


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




![群二维码地址](http://ww1.sinaimg.cn/large/007daNw2ly1gqvw912ofij60fo0ludit02.jpg)


![过期点我](http://ww1.sinaimg.cn/large/007daNw2ly1gqvm5w0rjvj30fo0lu41c.jpg)
