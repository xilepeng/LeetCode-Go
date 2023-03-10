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
4. [二分模板](#二分模板)
5. [模板一](#模板一)
6. [模板二](#模板二)

------


**Go 二分查找高级模版**

```go
func binarySearchLeft(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := low + (high-low)>>1
		if nums[mid] > target { // 答案在 mid 左侧 
			high = mid          // [low,mid] [mid+1,high]
		} else {
			low = mid + 1
		}
	}
	return nums[low] // low == high
}

func binarySearchRight(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := low + (high-low+1)>>1
		if nums[mid] <= target { // 答案在 mid 右侧 
			low = mid            // [low,mid-1] [mid,high]
		} else {
			high = mid - 1
		}
	}
	return nums[low] // low == high
}
```



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


``` go
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

``` go
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

``` go
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

``` go
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


``` go
func mySqrt(x int) int {
	low, high := 0, x
	for low < high {
		pivot := low + (high-low+1)>>1
		if pivot <= x/pivot {
			low = pivot // 答案在 pivot 右侧
		} else {
			high = pivot - 1
		}
	}
	return low
}
```

### 方法一：二分查找

``` go
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

``` go
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





## 二分模板

算法思路：假设目标值在闭区间[l, r]中，每次将区间长度缩小一半，当l = r 时，就找到了目标值。

![屏幕快照 2019-07-15 上午10.11.49.png](https://cdn.acwing.com/media/article/image/2019/07/15/2569_3b273314a6-屏幕快照-2019-07-15-上午10.11.49.png) 
## 模板一

mid = l + r >> 1， （下取整）
if mid是绿色,分界点target在mid左侧，可能包括mid, [l,r]->[l,mid],  r = mid
else mid是红色，分界点target在mid右侧，不包括mid , [l,r]->[mid+1,r], l = mid + 1

当我们将区间[l, r]划分成[l, mid], [mid + 1， r]时，其更新操作是 r = mid 或者 l = mid + 1。

C++代码模板：

``` go
    int bsearch_1(int l, int r)
    {
        while(l < r)
        {
            int mid = l + r >> 1;
            if(check(mid)) r = mid;
            else l = mid + 1;
        }
        return l;
    }
```
     
![屏幕快照 2019-07-15 上午10.11.56.png](https://cdn.acwing.com/media/article/image/2019/07/15/2569_52c8c0c8a6-屏幕快照-2019-07-15-上午10.11.56.png)  

## 模板二

mid = l + r +1 >> 1，（上取整）
if mid 是红色，分界点target在mid 右侧，可能包括mid, [l,r]->[mid,r], l = mid 
else mid 是绿色， 分界点target在mid 左侧， 不包括mid, [l,r]->[l,mid - 1] r = mid - 1

    注意：
    如果模板二用mid = l + r >> 1，（下取整）
    当l = r - 1， mid = l + r >>1 == 2l + 1 >>1  ==  l
    if mid 是红色，[l,r]->[mid,r]  ->[l,r]，死循环。
    
当我们将区间[l, r]划分成[l, mid - 1] 和[mid, r]时，更新操作是 r = mid - 1或者 l = mid,此时为了防止死循环，计算mid 时要加1。

C++代码模板：
``` go
    int bsearch_2(int l, int r)
    {
        while(l < r)
        {
            int mid = l + r + 1 >> 1;
            if(check(mid)) l = mid;
            else r = mid - 1;
        }
        return l;
    }
```



``` go
// 整数二分算法模板

bool check(int x) {/* ... */} // 检查x是否满足某种性质


// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}


// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}

```