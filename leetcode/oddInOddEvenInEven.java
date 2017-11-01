package leetcodeForJianzhi;

public class oddInOddEvenInEven {
	public static void oddInOddEvenInEven(int[] nums){
		int i=0,j=1;
		int lens=nums.length;
		while((i<lens)&&(j<lens)){
			while((i<lens)&& (nums[i]%2==0)){
				i+=2;
			}
			while((j<lens)&&(nums[j]%2==1)){
				j+=2;
			}
			if((i<lens)&&(j<lens)){
				swap(nums,i,j);
				i+=2;
				j+=2;
			}
		}
	}
	public static void swap(int[] nums,int start,int end){
		int temp=nums[start];
		nums[start]=nums[end];
		nums[end]=temp;
	}
	public static void main(String[] args) {
		int[] in={1,2,3,4};
		oddInOddEvenInEven(in);
		for (int i : in) {
			System.out.println(i);
		}
	}
}
