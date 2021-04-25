
[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)
[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

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