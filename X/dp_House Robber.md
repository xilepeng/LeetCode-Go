
[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

[337.打家劫舍III](https://leetcode-cn.com/problems/house-robber-iii/)

---

[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

There is some frustration when people publish their perfect fine-grained algorithms without sharing any information abut how they were derived. This is an attempt to change the situation. There is not much more explanation but it's rather an example of higher level improvements. Converting a solution to the next step shouldn't be as hard as attempting to come up with perfect algorithm at first attempt.

This particular problem and most of others can be approached using the following sequence:

    1. Find recursive relation
    2. Recursive (top-down)
    3. Recursive + memo (top-down)
    4. Iterative + memo (bottom-up)
    5. Iterative + N variables (bottom-up)

#### Step 1. Figure out recursive relation.
A robber has 2 options: a) rob current house i; b) don't rob current house.
If an option "a" is selected it means she can't rob previous i-1 house but can safely proceed to the one before previous i-2 and gets all cumulative loot that follows.
If an option "b" is selected the robber gets all the possible loot from robbery of i-1 and all the following buildings.
So it boils down to calculating what is more profitable:

- robbery of current house + loot from houses before the previous
- loot from the previous house robbery and any loot captured before that

rob(i) = Math.max( rob(i - 2) + currentHouseValue, rob(i - 1) )

#### Step 2. Recursive (top-down)
Converting the recurrent relation from Step 1 shound't be very hard.

```go
func rob(nums []int) int {
	return Rob(nums, len(nums)-1)
}
func Rob(nums []int, i int) int {
	if i < 0 {
		return 0
	}
	return max(Rob(nums, i-2)+nums[i], Rob(nums, i-1)) //max(抢 去下下家，不抢 去下家)
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```
Time Limit Exceeded
55/68 cases passed (N/A)

This algorithm will process the same i multiple times and it needs improvement. Time complexity: [to fill]


#### Step 3. Recursive + memo (top-down).

```go
var memo []int

func rob(nums []int) int {
	n := len(nums)
	memo = make([]int, n+1)
	for i := 0; i <= len(nums); i++ {
		memo[i] = -1
	}
	return Rob(nums, n-1)
}
func Rob(nums []int, i int) int {
	if i < 0 {
		return 0
	}
	if memo[i] >= 0 {
		return memo[i]
	}
	result := max(Rob(nums, i-2)+nums[i], Rob(nums, i-1)) //max(抢 去下下家，不抢 去下家)
	memo[i] = result
	return result
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


Much better, this should run in O(n) time. Space complexity is O(n) as well, because of the recursion stack, let's try to get rid of it.


#### Step 4. Iterative + memo (bottom-up)

```go
func rob(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	memo := make([]int, n+1)
	memo[0], memo[1] = 0, nums[0]
	for i := 1; i < n; i++ {
		val := nums[i]
		memo[i+1] = max(memo[i], memo[i-1]+val)
	}
	return memo[n]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

#### Step 5. Iterative + 2 variables (bottom-up)
We can notice that in the previous step we use only memo[i] and memo[i-1], so going just 2 steps back. We can hold them in 2 variables instead. This optimization is met in Fibonacci sequence creation and some other problems [to paste links].

```go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	prev1, prev2 := 0, 0
	for _, num := range nums {
		tmp := prev1
		prev1 = max(prev2+num, prev1)
		prev2 = tmp
	}
	return prev1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

```

[From good to great. How to approach most of DP problems.](https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.)

------


[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)



[213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

[337.打家劫舍III](https://leetcode-cn.com/problems/house-robber-iii/)


