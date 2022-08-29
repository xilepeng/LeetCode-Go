# 目录
# 二分查找

[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

[35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

[278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

[275. H 指数 II](https://leetcode-cn.com/problems/h-index-ii/)

#### 补充：
------
[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/) (百度)

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

[785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/)




------

3、为什么 left = mid + 1，right = mid ？和之前的算法不一样？
答：这个很好解释，因为我们的「搜索区间」是 [left, right) 左闭右开，所以当 nums[mid] 被检测之后，下一步的搜索区间应该去掉 mid 分割成两个区间，即 [left, mid) 或 [mid + 1, right)。

4、为什么该算法能够搜索左侧边界？
答：关键在于对于 nums[mid] == target 这种情况的处理：
``` go
    if nums[mid] == target {
		right = mid
	}     
```
可见，找到 target 时不要立即返回，而是缩小「搜索区间」的上界 right，在区间 [left, mid) 中继续搜索，即不断向左收缩，达到锁定左侧边界的目的。
5、为什么返回 left 而不是 right？
答：都是一样的，因为 while 终止的条件是 left == right。

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)




[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/) (百度)



[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

``` go
func search(nums []int, target int) int {
	if len(nums) == 0 {
		return -1
	}
	l, r := 0, len(nums)-1
	for l <= r {
		mid := (l + r) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[0] <= nums[mid] { //左边单调递增
			if nums[0] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[len(nums)-1] {
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return -1
}
```

[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

``` go
func mySqrt(x int) int {
	l, r := 0, x
	ans := -1
	for l <= r {
		mid := l + (r-l)/2
		if mid*mid <= x {
			ans = mid
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return ans
}
```
