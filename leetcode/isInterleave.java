package leetcodeForJianzhi;

public class isInterleave {
	/*题目描述

	Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
	For example,
	Given:
	s1 ="aabcc",
	s2 ="dbbca",
	When s3 ="aadbbcbcac", return true.
	When s3 ="aadbbbaccc", return false.*/
	
//首先解决第一行和第一列， （存在的意义是保证首先是s1或者s2是否是开始就等于s3，然后向下如果上面是true，那么下面的那个就跟着左边判断是否相等
//	如果左边是true就判断是否跟上面的相等）	
	public static  boolean isInterleave(String s1, String s2, String s3) {
		if(s1.length()+s2.length() !=s3.length()) return false;
		boolean[][] dp=new boolean[s1.length()+1][s2.length()+1];
		dp[0][0]=true;
		for(int i=1;i<=s1.length();i++){
			dp[i][0]=dp[i-1][0] && s1.charAt(i-1)==s3.charAt(i-1);
		}
		for(int j=1;j<=s2.length();j++){
			dp[0][j]=dp[0][j-1]&& s2.charAt(j-1)==s3.charAt(j-1);
		}
		for(int i=1;i<=s1.length();i++){
			for(int j=1;j<=s2.length();j++){
				dp[i][j]=((dp[i-1][j] && s1.charAt(i-1)==s3.charAt(i+j-1))||(dp[i][j-1] && s2.charAt(j-1)==s3.charAt(i+j-1)));
			}
		}
		return dp[s1.length()][s2.length()];
	}
	public static void main(String[] args) {
		String s1="aa";
		String s2="ab";
		String s3="aaba";
		System.out.println(isInterleave(s1, s2, s3));
	}
}
