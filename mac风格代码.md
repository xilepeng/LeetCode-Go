
## mac风格代码


<html>
<style>
    .mac {
        width:10px;
        height:10px;
        border-radius:5px;
        float:left;
        margin:10px 0 0 5px;
    }
    .b1 {
        background:#E0443E;
        margin-left: 10px;
    }
    .b2 { background:#DEA123; }
    .b3 { background:#1AAB29; }
    .warpper{
        /* background:#121212; */
        border-radius:5px;
        width:auto;
    }
</style>
<div class="warpper">
    <div class="mac b1"></div>
    <div class="mac b2"></div>
    <div class="mac b3"></div>
<div>
<br>
</html>


``` go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}

func quickSort(nums []int, start, end int) {
	if start >= end { // 子数组长度为 1 时终止递归
		return
	}
	pivot := nums[start+(end-start)>>1] // 选取中值 pivot 划分
	i, j := start-1, end+1
	for i < j {
		for i++; nums[i] < pivot; i++ { // 从左向右扫描，找到大于 pivot 的数，停止
		}
		for j--; nums[j] > pivot; j-- { // 从右向左扫描，找到小于 pivot 的数，停止
		}
		if i < j {
			nums[i], nums[j] = nums[j], nums[i] // 交换, 使得左边小于 pivot, 右边大于 pivot
		}
	}
	quickSort(nums, start, j) // 递归处理左边
	quickSort(nums, j+1, end) // 递归处理左边
}
```