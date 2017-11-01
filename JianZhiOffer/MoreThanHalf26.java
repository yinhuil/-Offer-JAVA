package JianZhiOffer;

public class MoreThanHalf26 {
/*	数组中有一个数字出现的次数超过数组长度的一半，
	请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
	由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。*/
//本题的依据是，当这个值的次数超过一半的时候，也就是这个值的次数，比其他所有值的个数和都多，所以，当下一个值如果不等于
//  当前值的话，就把个数减一，直到个数变为0位置，这个时候就将新的数值赋给count为0的值。
	public static int moreThanHalf(int[] nums){
		if(nums==null) return Integer.MAX_VALUE;
		int result=nums[0];
		int count=1;
		for(int i=1;i<nums.length;i++){
			if(nums[i]==result){
				count++;
			}else{
				if(count>0){
					count--;
				}else{
					result=nums[i];
					count=1;
				}
			}
		}
		count=0;
		for(int j=0;j<nums.length;j++){
			if(nums[j]==result) count++;
		}
		if(count>(nums.length>>1)){
			return result;
		}else{
			return Integer.MAX_VALUE;
		}
	}
//	下面采用的是第二种方法，找到中间的大小位于中间的那个数，然后来判断这个数的个数是不是整体数组的一半，如果是的话，
//	那就输出结果，不是的话 就输出0
//	因为如果一个数在整个数组当中超过一半，那么他肯定大小的位置位于中间
	public int MoreThanHalfNum_Solution(int[] array) {
		if(array.length==0) return 0;
		if(array.length==1) return array[0];
		int index=array.length>>1;
		int topk=findK(array, index+1, 0, array.length-1);
		int sum=0;
		for(int i=0;i<array.length;i++){
			if(array[i]==topk){
				sum+=1;
			}
		}
		if(sum>index){
			return topk;
		}else{
			return 0;
		}
	}
	private static int findK(int[] nums, int k, int start, int end) {
		int parti=nums[start],i=start,m=start;
		for(int j=start+1;j<=end;j++){
			if(nums[j]>parti){
				continue;
			}else{
				swap(nums,++i,j);
				if(parti!=nums[j]){
					swap(nums,m++,i);
				}
			}
		}
		if(k<=i && k>=m){
			return nums[k];
		}else if (k>i){
			return findK(nums,k,i+1,end);
		}else{
			return findK(nums,k,start,m-1);
		}
	}
	private static void swap(int[] nums, int a, int b) {
		int temp = nums[a];
		nums[a] = nums[b];
		nums[b] = temp;
	}
	public static void main(String[] args) {
		
		int[] in={1,2,3,2,2,2,5,4,2};
		System.out.println(moreThanHalf(in));
	}

}
