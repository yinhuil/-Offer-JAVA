package JianZhiOffer;
/*
 * 本题采用快排的思想：将第一个数字k作为基数，把小于k的值放到数组的左边，把大于k的值放在数组的右边（然后使用递归的方法，接着处理左边的数据和右边的数据），最后判断这个k是否是处于数数组的哪一位，如果是
 * 要求的第l位的数据，那么最后的结果就是nums[k](因为通过快排以后，k位的数据是整个数组的真实排序位置)
 * */
public class findKthLargest {
	public static int findKthLargest(int[] nums, int k) {
		return findK(nums, nums.length - k, 0, nums.length - 1);
	}

	private static int findK(int[] nums, int k, int start, int end) {
		int parti = nums[start], i = start, m = start;
		for (int j = start + 1; j <= end; j++) {
			if (nums[j] > parti)
				continue;
			if (nums[j] <= parti) {
				swap(nums, ++i, j);
				if (nums[j] != parti)
					swap(nums, m++, i);
			}
		}
		if (k >= m && k <= i)
			return nums[k];
		else if (k < m)
			return findK(nums, k, start, m - 1);
		else
			return findK(nums, k, i + 1, end);
	}
	private static void swap(int[] nums, int a, int b) {
		int temp = nums[a];
		nums[a] = nums[b];
		nums[b] = temp;
	}
	public static void main(String[] args) {
		int[] in={3,2,7,5,1,2};
		System.out.println(findKthLargest(in, 2));
	}
}
