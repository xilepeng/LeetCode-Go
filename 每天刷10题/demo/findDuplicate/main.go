package main

import "fmt"

func findDuplicate(nums []int) int {
	slow, fast := 0, 0
	for {
		slow, fast = nums[slow], nums[nums[fast]] // slow 跳1步，fast 跳2步
		if slow == fast {                         // 指针在环中首次相遇
			fast = 0 // 让快指针回到起点
			for {
				if slow == fast { // 指针在入口处再次相遇
					return slow // 返回入口，即重复数
				}
				slow, fast = nums[slow], nums[fast] // 两个指针每次都跳1步
			}
		}
	}
}

func main() {
	nums := []int{4, 3, 1, 2, 2}
	res := findDuplicate(nums)
	fmt.Println("重复数", res)
}

// func findDuplicate(nums []int) int {
// 	slow := nums[0]
// 	fast := nums[nums[0]]
// 	for slow != fast {
// 		slow = nums[slow]
// 		fast = nums[nums[fast]]
// 	}
// 	slow = 0
// 	for slow != fast {
// 		slow = nums[slow]
// 		fast = nums[fast]
// 	}
// 	return slow
// }
