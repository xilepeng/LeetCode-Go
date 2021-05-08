package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func merge_sort(A []int, start, end int) {
	if start < end {
		mid := (start + end) >> 1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	tmp := []int{}
	i, j := start, mid+1
	for i <= mid || j <= end {
		if i > mid || j <= end && A[j] < A[i] {
			tmp = append(tmp, A[j])
			j++
			count += mid - i + 1 //逆序对
		} else {
			tmp = append(tmp, A[i])
			i++
		}
	}
	copy(A[start:end+1], tmp)
}

var count, n int

func main() {
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	A := make([]int, n)
	num, i := 0, 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	merge_sort(A, 0, len(A)-1)
	fmt.Println(count)
}

/*


package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func merge_sort(A []int, start, end int) {
	if start < end {
		mid := (start + end) >> 1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	tmp := []int{}
	i, j := start, mid+1
	for i <= mid || j <= end {
		if i > mid || j <= end && A[j] < A[i] {
			tmp = append(tmp, A[j])
			j++
			if i <= mid {
				count += mid - i + 1 //逆序对
			}
		} else {
			tmp = append(tmp, A[i])
			i++
		}
	}
	copy(A[start:end+1], tmp)
}

var count int

func main() {
	// var n int
	// fmt.Scanf("%d", &n)
	// A := make([]int, n)
	// for i := 0; i < n; i++ {
	// 	fmt.Scanf("%d", &A[i])
	// }
	var n int
	_, _ = fmt.Scanln(&n)
	b, _ := ioutil.ReadAll(os.Stdin)
	// fmt.Println(b)
	i := 0
	A := make([]int, n)
	num := 0
	for _, by := range b {
		if by == ' ' {
			A[i] = num
			num = 0
			i++
		} else {
			num = num*10 + int(by-'0')
		}
	}
	A[i] = num

	merge_sort(A, 0, len(A)-1)
	fmt.Println(count)
}


*/
