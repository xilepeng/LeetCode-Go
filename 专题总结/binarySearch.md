<!-- 
[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)  补充

[69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/) -->


1. [704. 二分查找](#704-二分查找)
		1. [算法：](#算法)
2. [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置)
	1. [方法一：二分查找](#方法一二分查找)
	2. [解题思路](#解题思路)
	3. [方法二：二分查找](#方法二二分查找)
3. [69. x 的平方根](#69-x-的平方根)
	1. [方法一：二分查找](#方法一二分查找-1)
	2. [方法二：牛顿迭代](#方法二牛顿迭代)

------






## [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

1. 如果目标值等于中间元素，则找到目标值。
2. 如果目标值较小，继续在左侧搜索。
3. 如果目标值较大，则继续在右侧搜索。

#### 算法：

- 初始化指针 left = 0, right = n - 1。
- 当 left <= right：
比较中间元素 nums[mid] 和目标值 target 。
	1. 如果 target = nums[mid]，返回 mid。
	2. 如果 target < nums[mid]，则在左侧继续搜索 right = mid - 1。
	3. 如果 target > nums[mid]，则在右侧继续搜索 left = mid + 1。


```go
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
复杂度分析

- 时间复杂度：O(logN)。
- 空间复杂度：O(1)。



## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)



**方法一**

```go
func searchRange(nums []int, target int) []int {
	first, last := findFirst(nums, target), findLast(nums, target)
	return []int{first, last}
}
func findFirst(nums []int, target int) int {
	low, high := 0, len(nums)-1
	index := -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] >= target {
			high = mid - 1
		} else {
			low = mid + 1
		}
		if nums[mid] == target {
			index = mid
		}
	}
	return index
}
func findLast(nums []int, target int) int {
	low, high := 0, len(nums)-1
	index := -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] <= target {
			low = mid + 1
		} else {
			high = mid - 1
		}
		if nums[mid] == target {
			index = mid
		}
	}
	return index
}
```


### 方法一：二分查找

### 解题思路 
- 给出一个有序数组 nums 和一个数 target，要求在数组中找到第一个和这个元素相等的元素下标，最后一个和这个元素相等的元素下标。

- 这一题是经典的二分搜索变种题。二分搜索有 4 大基础变种题：

	1. 查找第一个值等于给定值的元素
	2. 查找最后一个值等于给定值的元素
	3. 查找第一个大于等于给定值的元素
	4. 查找最后一个小于等于给定值的元素
这一题的解题思路可以分别利用变种 1 和变种 2 的解法就可以做出此题。或者用一次变种 1 的方法，然后循环往后找到最后一个与给定值相等的元素。不过后者这种方法可能会使时间复杂度下降到 O(n)，因为有可能数组中 n 个元素都和给定元素相同。(4 大基础变种的实现见代码)

```go
func searchRange(nums []int, target int) []int {
	return []int{searchFirstEqualElement(nums, target), searchLastEqualElement(nums, target)}
}

// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func searchFirstEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == 0 || nums[mid-1] != target { // 找到第一个与 target 相等的元素
				return mid
			}
			right = mid - 1
		}
	}
	return -1
}

// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func searchLastEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == len(nums)-1 || nums[mid+1] != target { // 找到最后一个与 target 相等的元素
				return mid
			}
			left = mid + 1
		}
	}
	return -1
}

// 二分查找第一个大于等于 target 的元素，时间复杂度 O(logn)
func searchFirstGreaterElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] >= target {
			if mid == 0 || nums[mid-1] < target { // 找到第一个大于等于 target 的元素
				return mid
			}
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return -1
}

// 二分查找最后一个小于等于 target 的元素，时间复杂度 O(logn)
func searchLastLessElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] <= target {
			if mid == len(nums)-1 || nums[mid+1] > target { // 找到最后一个小于等于 target 的元素
				return mid
			}
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
### 方法二：二分查找

```go
func searchRange(nums []int, target int) []int {
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}
```








## [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

### 方法一：二分查找

```go
func mySqrt(x int) (res int) {
	left, right := 0, x
	for left <= right {
		mid := left + (right-left)>>1
		if mid*mid <= x {
			res = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return 
}
```
复杂度分析

- 时间复杂度：O(logx)，即为二分查找需要的次数。
- 空间复杂度：O(1)。

### 方法二：牛顿迭代

```go
func mySqrt(x int) int {
	r := x
	for r*r > x {
		r = (r + x/r) >> 1
	}
	return r
}
```
复杂度分析

- 时间复杂度：O(logx)，此方法是二次收敛的，相较于二分查找更快。
- 空间复杂度：O(1)。

