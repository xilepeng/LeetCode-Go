package main

import (
	"fmt"
	"math/rand"
	"time"
)

func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start]
	return i - 1
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().UnixNano())
	random := rand.Int()%(end-start+1) + start
	A[random], A[start] = A[start], A[random]
	return partition(A, start, end)
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
	}
	quick_sort(A, 0, n-1)
	for i := 0; i < n; i++ {
		fmt.Printf("%d ", A[i])
	}
}
