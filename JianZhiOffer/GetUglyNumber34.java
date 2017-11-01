package JianZhiOffer;

public class GetUglyNumber34 {
//我们把只包括因为2,3,5的数称为丑数
//	使用2,3,5这三个指针，来按照大小进行排列，找到第k个丑数。
	public static int getUglyNumber(int index){
		if(index<=0) return 0;
		int[] dp=new int[index];
		dp[0]=1;
		int index2=0,index3=0,index5=0,indexNumber=1;
		while(indexNumber<index){
			dp[indexNumber]=Math.min(Math.min(dp[index2]*2, dp[index3]*3), dp[index5]*5);
			while(dp[indexNumber]>=dp[index2]*2){
				index2++;
			}
			while(dp[indexNumber]>=dp[index3]*3){
				index3++;
			}
			while(dp[indexNumber]>=dp[index5]*5){
				index5++;
			}
			indexNumber++;
		}
		return dp[index-1];
	}
	public static void main(String[] args) {
		System.out.println(getUglyNumber(6));

	}

}
