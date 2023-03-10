
[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)


[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)


------


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)




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
``` go
func fib(n int) int {
	if n < 2 {
		return n 
	}
	p, q, r := 0, 0, 1
	for i := 2; i <= n; i++ {
		p = q
		q = r
		r = p + q
	}
	return r
}
```


复杂度分析

- 时间复杂度：O(n)。
- 空间复杂度：O(1)。

``` go
func fib(n int) int {
	a, b := 0, 1
	for i := 1; i <= n; i++ {
		c := a + b
		a = b
		b = c
	}
	return a
}
```

``` go
func fib(n int) int {
	a, b := 0, 1
	for i := 1; i <= n; i++ {
		a, b = b, (a + b)
	}
	return a
}
```


[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

``` go
func fib(n int) int {
    if n == 0 || n == 1 {
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        next := (prev + curr)%1000000007
        prev = curr
        curr = next
    }
    return curr
}
```

``` go
func fib(n int) int {
    a, b := 0, 1
    for i := 0; i < n; i++ {
        a, b = b, (a+b)%1000000007
    }
    return a
}
```



[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

``` go
func climbStairs(n int) int {
	prev, curr := 1, 1
	for i := 2; i <= n; i++ {
		next := prev + curr
		prev = curr
		curr = next
	}
	return curr
}
```

``` go
func climbStairs(n int) int {
	a, b := 1, 1
	for i := 0; i < n; i++ {
		a, b = b, a+b
	}
	return a
}
```



[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

``` go
func numWays(n int) int {
    prev, curr := 1, 1
    for i := 2; i <= n; i++ {
        next := (prev+curr)%1000000007
        prev = curr
        curr = next
    }
    return curr
}
```


``` go
func numWays(n int) int {
	a, b := 1, 1
	for i := 0; i < n; i++ {
		a, b = b, (a+b)%1000000007
	}
	return a
}
```









------


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

### 方法一：递归

![](images/509-1.png)

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

![](images/509-1-1.png)


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

![](images/509-2.png)
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

![](images/509-3.png)

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