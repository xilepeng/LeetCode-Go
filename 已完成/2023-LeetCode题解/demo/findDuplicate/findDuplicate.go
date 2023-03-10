package main

import "fmt"

func findDuplicate(nums []int) int {
	res := 0
	for i := 0; i < len(nums); i++ {
		res ^= nums[i] // 去重
	}
	for i := 1; i < len(nums); i++ {
		res ^= i // 二次去重
	}
	return res
}

func main() {
	nums := []int{1, 3, 4, 2, 5, 3}
	fmt.Println("输入数据：", nums)
	Duplicate := findDuplicate(nums)
	fmt.Println("找到唯一重复元素是：", Duplicate)
}

// 输入数据： [1 3 4 2 5 3]
// 找到唯一重复元素是： 3
