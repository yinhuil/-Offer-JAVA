package JianZhiOffer;

public class findGreatSumOfSubarray31 {
//当我此时的和是正数时，无论下一个数时什么都可以加在这个和上面（同时需要将此时的和放在最大值中），但此时的和是复数的时候，
//	就讲指针针对下一个数值同样进行存储最大的值。
	public static int findGreatSumOfSubarray1(int[] data){
		if(data==null) return 0;
		int total=data[0],maxsum=data[0];
		for(int i=1;i<data.length;i++){
			if(total>=0){
				total+=data[i];
			}else{
				total=data[i];
			}
			maxsum=Math.max(maxsum, total);
		}
		return maxsum;
	}
	public static int findGreatSumOfSubarray(int[] data){
		int lens=data.length;
		int[] dp=new int[lens];
		for(int i=0;i<lens;i++){
			if(i==0 || dp[i-1]<=0){
				dp[i]=data[i];
			}else{
				dp[i]=dp[i-1]+data[i];
			}
		}
		int sum=dp[0];
		for(int i=1;i<dp.length;i++){
			sum=Math.max(sum, dp[i]);
		}
		return sum;
		
	}
	public static void main(String[] args) {
		int[] in={1,-2,3,10,-15,7,2,-5};
		System.out.println(findGreatSumOfSubarray(in));

	}

}
