[460. LFU 缓存](https://leetcode-cn.com/problems/lfu-cache/)

[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)



[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) 


------
[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```go
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


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

### 方法一：递归

![截屏2021-04-22 11.34.30.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpsd9a3yaxj317o0o0q7h.jpg)

```go
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
```go
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

```go
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

```go
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

```go
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









# 买卖股票的最佳时机

**我们要跳出固有的思维模式，并不是要考虑买还是卖，而是要最大化手里持有的钱。
买股票手里的钱减少，卖股票手里的钱增加，无论什么时刻，我们要保证手里的钱最多。
并且我们这一次买还是卖只跟上一次我们卖还是买的状态有关。**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```go
func maxProfit(prices []int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, 0-p)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)


```go
func maxProfit(prices []int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

这两个问题唯一的不同点在于我们是买一次还是买无穷多次，而代码就只有 0-p 和 sell-p 的区别。
因为如果买无穷多次，就需要上一次卖完的状态。如果只买一次，那么上一个状态一定是0。

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)


第三题只允许最多买两次，那么就有四个状态，第一次买，第一次卖，第二次买，第二次卖。
还是那句话，无论什么状态，我们要保证手里的钱最多。

```go
func maxProfit(prices []int) int {
	b1, b2, s1, s2 := math.MinInt64, math.MinInt64, 0, 0
	for _, p := range prices {
		b1 = max(b1, 0-p)
		s1 = max(s1, b1+p)
		b2 = max(b2, s1-p)
		s2 = max(s2, b2+p)
	}
	return s2
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```go
func maxProfit(k int, prices []int) int {
	if k >= len(prices)>>1 {
		T_ik0, T_ik1 := 0, math.MinInt64
		for _, price := range prices {
			T_ik0_old := T_ik0
			T_ik0 = max(T_ik0, T_ik1+price)
			T_ik1 = max(T_ik1, T_ik0_old-price)
		}
		return T_ik0
	}
	T_ik0, T_ik1 := make([]int, k+1), make([]int, k+1)
	for i := range T_ik0 {
		T_ik0[i] = 0
		T_ik1[i] = math.MinInt64
	}
	for _, price := range prices {
		for j := k; j > 0; j-- {
			T_ik0[j] = max(T_ik0[j], T_ik1[j]+price)
			T_ik1[j] = max(T_ik1[j], T_ik0[j-1]-price)
		}
	}
	return T_ik0[k]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

这道题只是第二题的变形，卖完要隔一天才能买，那么就多记录前一天卖的状态即可。

```go
func maxProfit(prices []int) int {
	buy, sell_pre, sell := math.MinInt64, 0, 0
	for _, p := range prices {
		buy = max(buy, sell_pre-p)
		sell_pre, sell = sell, max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

每次买卖需要手续费，那么我们买的时候减掉手续费就行了。
```go
func maxProfit(prices []int, fee int) int {
	buy, sell := math.MinInt64, 0
	for _, p := range prices {
		buy = max(buy, sell-p-fee)
		sell = max(sell, buy+p)
	}
	return sell
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```





[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) 

![](https://pic.leetcode-cn.com/1605227711-BsDKjR-image.png)

思路
- odd 指针扫描奇数结点，even 指针扫描偶数结点

	- 奇数结点逐个改 next，连成奇链
	- 偶数结点逐个改 next，连成偶链
- 循环体内，做 4 件事：

	- 当前奇数结点 ——next——> 下一个奇数结点
	- odd 指针推进 ——————> 下一个奇数结点
	- 当前偶数结点 ——next——> 下一个偶数结点
	- even 指针推进 ——————> 下一个偶数结点

- 扫描结束时，奇链偶链就分开了，此时 odd 指向奇链的尾结点
- 奇链的尾结点 ——next——> 偶链的头结点（循环前保存），就连接了奇偶链

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func oddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	evenHead := head.Next
	odd, even := head, evenHead
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = evenHead
	return head
}
```


[7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)




[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)



[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	merge_sort(nums, 0, len(nums)-1)
	return nums
}
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] < A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```