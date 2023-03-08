
1. [13. 找出数组中重复的数字](#13-找出数组中重复的数字)
2. [14. 不修改数组找出重复的数字](#14-不修改数组找出重复的数字)



## [13. 找出数组中重复的数字](https://www.acwing.com/problem/content/14/)

```go
func duplicateInArray(nums []int) int {
    n := len(nums)
    for _,x := range nums {
        if x < 0 || x >= n {
            return -1
        }
    }
    for i := range nums {
        for nums[i] != nums[nums[i]] {
            nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
        }
        if nums[i] != i {
            return nums[i]
        }
    }
    return -1
}
    
```

## [14. 不修改数组找出重复的数字](https://www.acwing.com/problem/content/15/)

```go
func duplicateInArray(nums []int) int {
    l, r := 1, len(nums)-1
    for l < r {
        mid := (l+r)>>1 // 划分的区间：[l, mid], [mid + 1, r]
        s := 0
        for _,x := range nums {
            if x >= l && x <= mid {
                s++
            }
        }
        if s > mid-l+1 { // [l,mid] 区间个数大于区间长度，有重复
            r = mid
        } else {
            l = mid+1
        }
    }
    return r
}
```