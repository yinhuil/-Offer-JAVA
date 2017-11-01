package leetcodeForJianzhi;
/*题目描述

A message containing letters fromA-Zis being encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it.
For example,
Given encoded message"12", it could be decoded as"AB"(1 2) or"L"(12).
The number of ways decoding"12"is 2.*/

//题解：如果发现当前值为0的时候，就是首先设为0，因为他不能单个存在，只能看前面的，  如果不为0，那就可以作为单列存在等于dp[i+1]=dp[i]
// 然后在考虑两个数字在一块的情况
public class numDecodings {
    public static int numDecodings(String s) {
    	if(s.length()==0|| s==null) return 0; 
        int[] dp=new int[s.length()+1];
        dp[0]=1;
        for(int i=0;i<s.length();i++){
        	if(s.charAt(i)=='0'){
        		dp[i+1]=0;
        	}else{
        		dp[i+1]=dp[i];
        	}
        	if((i>=1 &&s.charAt(i-1)=='1')||(i>=1 && s.charAt(i-1)=='2' && s.charAt(i)<='6')){
        		dp[i+1]+=dp[i-1];
        	}
        }
        return dp[s.length()];
    }
    public static void main(String[] args) {
		String s="123";
		System.out.println(numDecodings(s));
	}
}
