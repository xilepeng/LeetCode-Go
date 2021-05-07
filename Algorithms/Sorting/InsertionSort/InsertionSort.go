package main

import (
	"fmt"
)

func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		tmp, j := A[i], i
		for j > 0 && tmp < A[j-1] {
			A[j] = A[j-1]
			j--
		}
		A[j] = tmp
	}
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A, B, hash := make([]int, n), make([]int, n), make(map[int]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		B[i] = A[i]
	}

	insertion_sort(A, n)
	for i := 0; i < n; i++ {
		hash[A[i]] = i + 1
	}
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", hash[B[i]])
	}
}

// input
// 5
// 9 7 8 12 10

// 7 8 9 10 12 -- sorted array
// output-- 3 1 2 5 4
