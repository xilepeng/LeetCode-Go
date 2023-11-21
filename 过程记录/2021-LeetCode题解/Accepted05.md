

1. [958. 二叉树的完全性检验](#958-二叉树的完全性检验)
	1. [方法一：广度优先搜索](#方法一广度优先搜索)
2. [322. 零钱兑换](#322-零钱兑换)
		1. [iterate amount](#iterate-amount)
3. [518. 零钱兑换 II 补充](#518-零钱兑换-ii-补充)
		1. [iterate coins](#iterate-coins)
4. [剑指 Offer 36. 二叉搜索树与双向链表](#剑指-offer-36-二叉搜索树与双向链表)
5. [179. 最大数](#179-最大数)
	1. [func Atoi](#func-atoi)
	2. [func Itoa](#func-itoa)
6. [剑指 Offer 10- I. 斐波那契数列](#剑指-offer-10--i-斐波那契数列)
7. [509. 斐波那契数 补充](#509-斐波那契数-补充)
	1. [方法一：递归](#方法一递归)
	2. [方法二：带备忘录递归](#方法二带备忘录递归)
	3. [方法三：动态规划](#方法三动态规划)
	4. [方法四：滚动数组](#方法四滚动数组)
8. [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](#剑指-offer-21-调整数组顺序使奇数位于偶数前面)
9. [剑指 Offer 54. 二叉搜索树的第k大节点](#剑指-offer-54-二叉搜索树的第k大节点)
10. [162. 寻找峰值](#162-寻找峰值)
	1. [方法一：二分查找](#方法一二分查找)
11. [876. 链表的中间结点 补充](#876-链表的中间结点-补充)
12. [24. 两两交换链表中的节点](#24-两两交换链表中的节点)
13. [14. 最长公共前缀](#14-最长公共前缀)
	1. [方法一：纵向扫描](#方法一纵向扫描)
14. [468. 验证IP地址](#468-验证ip地址)
15. [227. 基本计算器 II](#227-基本计算器-ii)
16. [43. 字符串相乘](#43-字符串相乘)
17. [补充题6. 手撕堆排序 912. 排序数组](#补充题6-手撕堆排序-912-排序数组)
18. [59. 螺旋矩阵 II](#59-螺旋矩阵-ii)
19. [498. 对角线遍历 next](#498-对角线遍历-next)
20. [7. 整数反转](#7-整数反转)
21. [32. 最长有效括号](#32-最长有效括号)
22. [128. 最长连续序列](#128-最长连续序列)
23. [283. 移动零](#283-移动零)


<!-- 
[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

[322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

[518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/) 补充

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

[179. 最大数](https://leetcode-cn.com/problems/largest-number/)

[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/) 补充

[剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

[剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

[24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

[14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

[468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)

[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

[43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

[498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/)

[7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

[128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

[283. 移动零](https://leetcode-cn.com/problems/move-zeroes/) -->


------

## [958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

### 方法一：广度优先搜索
1. 按 根左右(前序遍历) 顺序依次检查
2. 如果出现空节点，标记end = true
3. 如果后面还有节点，返回false

``` go
func isCompleteTree(root *TreeNode) bool {
	q, end := []*TreeNode{root}, false
	for len(q) > 0 {
		node := q[0]
		q = q[1:]
		if node == nil {
			end = true
		} else {
			if end == true {
				return false
			}
			q = append(q, node.Left)
			q = append(q, node.Right)
		}
	}
	return true
}
```


## [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)


![322. Coin Change and 518. Coin Change 2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gps6k2bgrtj31kg3tub29.jpg)

![截屏2021-04-23 16.55.43.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6iwvafj319i0o042z.jpg)

![截屏2021-04-23 13.16.57.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptltwipl8j319q0p2go8.jpg)


#### iterate amount

```go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1 // 初始化 dp 数组为最大值
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ { //遍历所有状态的所有值
		for _, coin := range coins {
			if i-coin >= 0 { //求所有选择的最小值 min(dp[4],dp[3],dp[0])+1
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```



## [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/) 补充

![截屏2021-04-23 16.57.11.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpts6y27nhj319a0n8gpb.jpg)

#### iterate coins

``` go
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```


## [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

``` go
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}
*/
var head, prev *TreeNode

func treeToDoublyList(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	dfs(root)
	head.Left, prev.Right = prev, head
	return head
}
func dfs(curr *TreeNode) {
	if curr == nil {
		return
	}
	dfs(curr.Left)
	if prev == nil {
		head = curr
	} else {
		prev.Right, curr.Left = curr, prev
	}
	prev = curr
	dfs(curr.Right)
}

```



## [179. 最大数](https://leetcode-cn.com/problems/largest-number/)

``` go
func largestNumber(nums []int) string {
	if len(nums) == 0 {
		return ""
	}
	numStrs := toStringArray(nums)
	quickSort(numStrs, 0, len(numStrs)-1)
	res := ""
	for _, str := range numStrs {
		if res == "0" && str == "0" {
			continue
		}
		res = res + str
	}
	return res
}
func toStringArray(nums []int) []string {
	strs := make([]string, 0)
	for _, num := range nums {
		strs = append(strs, strconv.Itoa(num))
	}
	return strs
}
func quickSort(A []string, start, end int) {
	if start < end {
		piv_pos := partition(A, start, end)
		quickSort(A, start, piv_pos-1)
		quickSort(A, piv_pos+1, end)
	}
}
func partition(A []string, start, end int) int {
	piv, i := A[start], start+1
	for j := start + 1; j <= end; j++ { //注意：必须加等号
		ajStr := A[j] + piv
		pivStr := piv + A[j]
		fmt.Printf("%v %v\n", ajStr, pivStr) //210 102
		if ajStr > pivStr {
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start]
	return i - 1
}
```



``` go
func largestNumber(nums []int) string {
	if len(nums) == 0 {
		return ""
	}
	strArr := toStringArray(nums)
	quickSort(strArr, 0, len(strArr)-1)
	res := ""
	for _, str := range strArr {
		if res == "0" && str == "0" {
			continue
		}
		res = res + str
	}
	return res
}
func toStringArray(nums []int) []string {
	str := []string{}
	for _, i := range nums {
		str = append(str, strconv.Itoa(i))
	}
	return str
}
func quickSort(A []string, start, end int) {
	if start < end {
		piv_pos := partition(A, start, end)
		quickSort(A, start, piv_pos-1)
		quickSort(A, piv_pos+1, end)
	}
}
func partition(A []string, start, end int) int {
	piv, i := A[end], start-1
	for j := start; j < end; j++ {
		ajPiv := A[j] + piv
		pivAj := piv + A[j]
		if ajPiv > pivAj {
			i++
			A[i], A[j] = A[j], A[i]
		}
	}
	A[i+1], A[end] = A[end], A[i+1]
	return i + 1
}
```





### func Atoi
``` go
func Atoi(s string) (i int, err error)
```
Atoi是ParseInt(s, 10, 0)的简写。


### func Itoa
``` go
func Itoa(i int) string
```
Itoa是FormatInt(i, 10) 的简写。

``` go
package main

import (
	"fmt"
	"strconv"
)

func main() {
	nums := []int{1, 2, 3}
	s := []string{}
	num := []int{}
	for _, n := range nums {
		s = append(s, strconv.Itoa(n))
	}
	fmt.Println("整数转字符串", s) //字符串 [1 2 3] 数据类型：string

	for i := 0; i < len(s); i++ {
		n, _ := strconv.Atoi(s[i])
		num = append(num, n)
	}
	fmt.Println("字符串转整数", num) //整数 [1 2 3]
}
```




## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

``` go
func fib(n int) int {
    if n < 2 { 
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        sum := prev + curr
        prev = curr
        curr = sum % 1000000007
    }
    return curr 
}
```

## [509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/) 补充

### 方法一：递归

![截屏2021-04-22 11.34.30.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsd9a3yaxj317o0o0q7h.jpg)

``` go
func fib(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	return fib(n-1) + fib(n-2)
}
```

复杂度分析

- 时间复杂度：O(2^n)。
- 空间复杂度：O(h)。

### 方法二：带备忘录递归


![截屏2021-04-23 20.07.36.png](http://ww1.sinaimg.cn/large/007daNw2ly1gptxon19akj319k0nw407.jpg)

闭包写法：
``` go
func fib(n int) int {
	memo := make([]int, n+1)//从0开始
	var helper func(int) int

	helper = func(n int) int {
		if n == 0 || n == 1 {
			return n
		}
		if memo[n] != 0 {
			return memo[n]
		}
		memo[n] = helper(n-1) + helper(n-2)
		return memo[n]
	}

	return helper(n)
}
```

``` go
func fib(n int) int {
	memo := make([]int, n+1)
	return helper(memo, n)
}
func helper(memo []int, n int) int {
	if n < 2 {
		return n
	}
	if memo[n] != 0 { //剪枝
		return memo[n]
	}
	memo[n] = helper(memo, n-1) + helper(memo, n-2)
	return memo[n]
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法三：动态规划

![截屏2021-04-22 11.35.07.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsda7wdjwj30zu0hwmyn.jpg)

``` go
func fib(n int) int {
	if n == 0 {
		return 0
	}
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 1       //base case
	for i := 2; i <= n; i++ { //状态转移
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
```
复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(n)。

### 方法四：滚动数组

![截屏2021-04-22 11.42.22.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsdgjxvorj31520kgjsx.jpg)

动态规划空间优化：只存储前2项

``` go
func fib(n int) int {
	if n == 0 || n == 1 { //base case
		return n
	} //递推关系
	prev, curr := 0, 1
	for i := 2; i <= n; i++ {
		next := prev + curr
		prev = curr
		curr = next
	}
	return curr
}
```

复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(1)。





## [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

``` go
func exchange(nums []int) []int {
    for i, j := 0, 0; i < len(nums); i++ {
        if nums[i] & 1 == 1 { //nums[i]奇数      nums[j]偶数
            nums[i], nums[j] = nums[j], nums[i] //奇偶交换
            j++
        }
    }
    return nums
}
```

``` go
func exchange(nums []int) []int {
    i, j := 0, len(nums)-1
    for i < j {
        for i < j && nums[i] & 1 == 1 { //从左往右、奇数一直向右走
            i++
        }
        for i < j && nums[j] & 1 == 0 { //从右向左、偶数一直向左走
            j--
        }
        nums[i], nums[j] = nums[j], nums[i] //奇偶交换
    }
    return nums
}
```




## [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

``` go
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

``` go
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





## [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

### 方法一：二分查找


``` go
func findPeakElement(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := low + (high-low)>>1   // (low + high) / 2
		if nums[mid] < nums[mid+1] { // 单调递增，mid 右边存在峰值
			low = mid + 1 
		} else {                     // 单调递减，mid 左边存在峰值
			high = mid 
		}
	}
	return low // low == high 
}
```




## [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head 
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}
```



## [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)



``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := head.Next
	head.Next = swapPairs(newHead.Next)
	newHead.Next = head
	return newHead
}
```



``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{0, head}
	temp := dummy
	for temp.Next != nil && temp.Next.Next != nil {
		node1 := temp.Next
		node2 := temp.Next.Next
		temp.Next = node2
		node1.Next = node2.Next
		node2.Next = node1
		temp = node1
	}
	return dummy.Next
}
```



## [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

### 方法一：纵向扫描

纵向扫描时，从前往后遍历所有字符串的每一列，比较相同列上的字符是否相同，如果相同则继续对下一列进行比较，如果不相同则当前列不再属于公共前缀，当前列之前的部分为最长公共前缀。

![](https://assets.leetcode-cn.com/solution-static/14/14_fig2.png)

``` go
func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		for j := 1; j < len(strs); j++ {
			if i == len(strs[j]) || strs[j][i] != strs[0][i] {
				return strs[0][:i]
			}
		}
	}
	return strs[0]
}
```


## [468. 验证IP地址](https://leetcode-cn.com/problems/validate-ip-address/)




``` go
func validIPAddress(IP string) string {
	if validIPv4Address(IP) {
		return "IPv4"
	}
	if validIPv6Address(IP) {
		return "IPv6"
	}
	return "Neither"
}

func validIPv4Address(IP string) bool {
	strArr := strings.Split(IP, ".")
	if len(strArr) != 4 {
		return false
	}
	for _, str := range strArr {
		num, err := strconv.Atoi(str)
		if err != nil || num > 255 || num < 0 { //注意：err != nil
			return false
		}
		newStr := fmt.Sprint(num)
		if str != newStr {
			return false
		}
	}
	return true
}

func validIPv6Address(IP string) bool {
	strArr := strings.Split(IP, ":")
	if len(strArr) != 8 {
		return false
	}
	for _, str := range strArr {
		if len(str) == 0 || len(str) > 4 {
			return false
		}
		for i := 0; i < len(str); i++ {
			if !(str[i] >= '0' && str[i] <= '9') &&
				!(str[i] >= 'a' && str[i] <= 'f') &&
				!(str[i] >= 'A' && str[i] <= 'F') {
				return false
			}
		}
	}
	return true
}
```

``` go
func validIPAddress(IP string) string {
	if validIPv4Address(IP) {
		return "IPv4"
	}
	if validIPv6Address(IP) {
		return "IPv6"
	}
	return "Neither"
}

func validIPv4Address(IP string) bool {
	strArr := strings.Split(IP, ".")
	if len(strArr) != 4 {
		return false
	}
	for _, str := range strArr {
		if num, err := strconv.Atoi(str); err != nil || num > 255 || num < 0 {
			return false
		} else if strconv.Itoa(num) != str {
			return false
		}
	}
	return true
}
func validIPv6Address(IP string) bool {
	strArr := strings.Split(IP, ":")
	if len(strArr) != 8 {
		return false
	}
	re := regexp.MustCompile(`^([0-9]|[a-f]|[A-F])+$`)
	for _, str := range strArr {
		if len(str) == 0 || len(str) > 4 {
			return false
		}
		if !re.MatchString(str) {
			return false
		}
	}
	return true
}
```






## [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

``` go
func calculate(s string) int {
	stack, sign, num, res := []int{}, byte('+'), 0, 0
	for i := 0; i < len(s); i++ {
		isDigit := s[i] <= '9' && s[i] >= '0'
		if isDigit {
			num = num*10 + int(s[i]-'0')
		}
		if !isDigit && s[i] != ' ' || i == len(s)-1 {
			if sign == '+' {
				stack = append(stack, num)
			} else if sign == '-' {
				stack = append(stack, -num)
			} else if sign == '*' {
				stack[len(stack)-1] *= num
			} else if sign == '/' {
				stack[len(stack)-1] /= num
			}
			sign = s[i]
			num = 0
		}
	}
	for _, i := range stack {
		res += i
	}
	return res
}
```

- 加号：将数字压入栈；
- 减号：将数字的相反数压入栈；
- 乘除号：计算数字与栈顶元素，并将栈顶元素替换为计算结果。

``` go
func calculate(s string) int {
	stack, preSign, num, res := []int{}, '+', 0, 0
	for i, ch := range s {
		isDigit := '0' <= ch && ch <= '9' // ch 是 0-9 的数字
		if isDigit {
			num = num*10 + int(ch-'0') //字符转数字
		}
		if !isDigit && ch != ' ' || i == len(s)-1 { // ch 是运算符
			switch preSign {
			case '+':
				stack = append(stack, num)
			case '-':
				stack = append(stack, -num)
			case '*':
				stack[len(stack)-1] *= num
			default:
				stack[len(stack)-1] /= num
			}
			preSign = ch
			num = 0
		}
	}
	for _, v := range stack {
		res += v
	}
	return res
}
```








## [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)



``` go
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1), len(num2)
	A := make([]int, m+n)
	for i := m - 1; i >= 0; i-- {
		x := int(num1[i]) - '0'
		for j := n - 1; j >= 0; j-- {
			y := int(num2[j]) - '0'
			A[i+j+1] += x * y
		}
	}
	for i := m + n - 1; i > 0; i-- { //进位
		A[i-1] += A[i] / 10
		A[i] %= 10
	}
	res, i := "", 0
	if A[0] == 0 {
		i = 1
	}
	for ; i < m+n; i++ {
		res += strconv.Itoa(A[i]) //整数转字符串、拼接
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(mn)
- 空间复杂度：O(m+n)


## [补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)


思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。


``` go
func sortArray(nums []int) []int {
	heapSort(nums)
	return nums
}
func heapSort(a []int) {
	heapSize := len(a)
	buildMaxHeap(a, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		a[0], a[i] = a[i], a[0] //堆顶元素和堆底元素交换
		heapSize--              //把剩余待排序元素整理成堆
		maxHeapify(a, 0, heapSize)
	}
}
func buildMaxHeap(a []int, heapSize int) { // O(n)
	for i := heapSize / 2; i >= 0; i-- { // 从后往前调整所有非叶子节点
		maxHeapify(a, i, heapSize)
	}
}
func maxHeapify(a []int, i, heapSize int) { // O(nlogn)
	l, r, largest := i*2+1, i*2+2, i
	if l < heapSize && a[largest] < a[l] { //左儿子存在且大于a[largest]
		largest = l
	}
	if r < heapSize && a[largest] < a[r] { //右儿子存在且大于a[largest]
		largest = r
	}
	if largest != i {
		a[largest], a[i] = a[i], a[largest] //堆顶调整为最大值
		maxHeapify(a, largest, heapSize)    //递归处理
	}
}
```

复杂度分析

- 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n−1 次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n−1 次调整即需要 O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(nlogn)。

- 空间复杂度：O(1)。只需要常数的空间存放若干变量。




## [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

``` go
func generateMatrix(n int) [][]int {
	matrix, num := make([][]int, n), 1
	for i := range matrix {
		matrix[i] = make([]int, n)
	}
	left, right, top, bottom := 0, n-1, 0, n-1
	for num <= n*n {
		for i := left; i <= right; i++ {
			matrix[top][i] = num
			num++
		}
		top++
		for i := top; i <= bottom; i++ {
			matrix[i][right] = num
			num++
		}
		right--
		for i := right; i >= left; i-- {
			matrix[bottom][i] = num
			num++
		}
		bottom--
		for i := bottom; i >= top; i-- {
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}
```
























## [498. 对角线遍历](https://leetcode-cn.com/problems/diagonal-traverse/) next

``` go
func findDiagonalOrder(mat [][]int) []int {
	res := []int{}
	if len(mat) == 0 {
		return res
	}
	i, j, m, n, up := 0, 0, len(mat), len(mat[0]), true
	for i < m && j < n {
		res = append(res, mat[i][j])
		if up {
			if j == n-1 { // 右边界
				i, up = i+1, false
			} else if i == 0 { // 上边界
				j, up = j+1, false
			} else {
				i, j = i-1, j+1
			}
		} else {
			if i == m-1 { // 下边界
				j, up = j+1, true
			} else if j == 0 { // 左边界
				i, up = i+1, true
			} else {
				i, j = i+1, j-1
			}
		}
	}
	return res
}
```


## [7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

``` go
func reverse(x int) int {
	res := 0
	for x != 0 {
		res = res*10 + x%10
		x /= 10
	}
	if res < -(1<<31) || res > 1<<31-1 {
		return 0
	}
	return res
}
```

## [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

``` go
func longestValidParentheses(s string) int {
	left, right, maxLength, n := 0, 0, 0, len(s)
	for i := 0; i < n; i++ {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			maxLength = max(maxLength, right*2)
		} else if left < right {
			left, right = 0, 0
		}
	}
	left, right = 0, 0
	for i := n - 1; i >= 0; i-- {
		if s[i] == ')' {
			right++
		} else {
			left++
		}
		if left == right {
			maxLength = max(maxLength, left*2)
		} else if left > right {
			left, right = 0, 0
		}
	}
	return maxLength
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

``` go
func longestValidParentheses(s string) int {
	stack, res := []int{-1}, 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				res = max(res, i-stack[len(stack)-1])
			}
		}
	}
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

## [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)


``` go
func longestConsecutive(nums []int) int {
	m, longgest := map[int]bool{}, 0
	for _, num := range nums {
		m[num] = true // 标记 nums 数组中所有元素都存在
	}
	for _, num := range nums {
		if !m[num-1] { // 要枚举的数 num 一定是在数组中不存在前驱数 num−1
			currNum, currLonggest := num, 1
			for m[currNum+1] { // 枚举数组中的每个数 x，考虑以其为起点，不断尝试匹配 x+1,x+2,⋯ 是否存在
				currNum++      // 枚举连续的下一个数
				currLonggest++ // 当前最长连续长度递增
			}
			if currLonggest > longgest {
				longgest = currLonggest
			}
		}
	}
	return longgest
}
```




## [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)


``` go
func moveZeroes(nums []int) {
	i, j, n := 0, 0, len(nums)
	for j < n {
		if nums[j] != 0 {
			nums[i], nums[j] = nums[j], nums[i]
			i++
		}
		j++
	}
}
```




