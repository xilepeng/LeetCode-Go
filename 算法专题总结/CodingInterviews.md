## 《剑指 Offer》



[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)



------




[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

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


[509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/) 补充

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



[剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

``` go
func reversePairs(nums []int) int {
    return merge_sort(nums, 0, len(nums)-1)
}
func merge_sort(A []int, start, end int) int {
    if start >= end { return 0}
    mid := start + (end-start)>>1
    cnt := merge_sort(A, start, mid) + merge_sort(A, mid+1, end)

    Arr := make([]int, end-start+1)
    p, q, k := start, mid+1, 0
    for i := start; i <= end; i++ {
        if p > mid {
            Arr[k] = A[q]
            q++
        } else if q > end {
            Arr[k] = A[p]
            p++
        } else if A[p] > A[q] {
            cnt += mid-p+1
            Arr[k] = A[q]
            q++
        } else {
            Arr[k] = A[p]
            p++
        }
        k++
    }
    copy(A[start:end+1], Arr)
    return cnt
}
```

``` go
func reversePairs(nums []int) int {
    return merge_sort(nums, 0, len(nums)-1)
}
func merge_sort(A []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	return merge_sort(A, start, mid) + merge_sort(A, mid+1, end) + merge(A, start, mid, end)
}
func merge(A []int, start, mid, end int) int {
	Arr, cnt := make([]int, end-start+1), 0
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] > A[q] {
			cnt += mid - p + 1
			Arr[k] = A[q]
			q++
		} else {
			Arr[k] = A[p]
			p++
		}
		k++
	}
	copy(A[start:end+1], Arr)
    return cnt
}
```

### 方法一：Merge Sort

``` go
func reversePairs(nums []int) int {
    n := len(nums)
    temp := make([]int, n)
    return merge_sort(nums, temp, 0, n-1)
}
func merge_sort(A, temp []int, start, end int) int {
    if start >= end {
        return 0
    }
    mid := start + (end - start)>>1
    count := merge_sort(A, temp, start, mid) + merge_sort(A, temp, mid+1, end)
    i, j, k := start, mid+1, 0
    for ; i <= mid && j <= end; k++ {
        if A[i] > A[j] {
            count += mid-i+1
            temp[k] = A[j]
            j++
        } else {
            temp[k] = A[i]
            i++
        }
    }
    for ; i <= mid; i++ {
        temp[k] = A[i]
        k++
    }
    for ; j <= end; j++ {
        temp[k] = A[j]
        k++
    }
    copy(A[start:end+1], temp)
    return count
}
```

### 方法二：Merge Sort

``` go
func reversePairs(nums []int) int {
    n := len(nums)
    Arr := make([]int, n)
    return merge_sort(nums,Arr, 0, n-1)
}
func merge_sort(A, Arr []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)>>1
	left := merge_sort(A, Arr, start, mid)
	right := merge_sort(A, Arr, mid+1, end)
	cross := merge(A, Arr, start, mid, end)
	return left + right + cross
}
func merge(A, Arr []int, start, mid, end int) int {
	p, q, k, count := start, mid+1, 0, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
			count += mid - p + 1
		}
		k++
	}
	for p = 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
	return count
}
```

### 方法三：Merge Sort

``` go
func reversePairs(nums []int) int {
    n := len(nums)
    return merge_sort(nums, 0, n-1)
}
func merge_sort(A []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := (start + end) >> 1
	left := merge_sort(A, start, mid)
	right := merge_sort(A, mid+1, end)
	cross := merge(A, start, mid, end)
	return left + right + cross
}
func merge(A []int, start, mid, end int) int {
	temp, count := []int{}, 0
	i, j := start, mid+1
	for i <= mid && j <= end {
		if A[i] > A[j] {
			count += mid - i + 1
			temp = append(temp, A[j])
			j++
		} else {
			temp = append(temp, A[i])
			i++
		}
	}
	for ; i <= mid; i++ {
		temp = append(temp, A[i])
	}
	for ; j <= end; j++ {
		temp = append(temp, A[j])
	}
	copy(A[start:end+1], temp)
	return count
}
```




[剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

``` go
package main

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

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

func main() {
	root := &TreeNode{4, nil, nil}
	node1 := &TreeNode{2, nil, nil}
	node2 := &TreeNode{5, nil, nil}
	node3 := &TreeNode{1, nil, nil}
	node4 := &TreeNode{3, nil, nil}
	root.Left = node1
	root.Right = node2
	node1.Left = node3
	node1.Right = node4
	head := treeToDoublyList(root)
	tail := head.Left
	//从头开始遍历
	for i := 0; i <= 9; i++ {
		fmt.Printf("%d\t", head.Val)
		head = head.Right
	}
	//从尾开始遍历
	for i := 0; i <= 9; i++ {
		fmt.Printf("%d\t", tail.Val)
		tail = tail.Left
	}

}


```