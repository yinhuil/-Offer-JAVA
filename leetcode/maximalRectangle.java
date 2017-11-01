package leetcodeForJianzhi;

import java.util.Stack;

/*题目描述

Given a 2D binary matrix filled with 0's and 1's, find the largest
rectangle containing all ones and return its area.*/
public class maximalRectangle {
	public static int maximalRectangle(char[][] matrix) {
		if(matrix.length<1 || matrix[0].length<1)
			return 0;
		int m=matrix.length;
		int n=matrix[0].length;
		int maxArea=0;
		int[] height=new int [n];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				if(matrix[i][j]=='1'){
					height[j]+=1;
				}else{
					height[j]=0;
				}
			}
			maxArea=Math.max(largestRecangleArea(height), maxArea);
			
		}
		return maxArea;
	}

	private static int largestRecangleArea(int[] height) {
		Stack<Integer> sta=new Stack<Integer>();
		int maxArea=0;
		int i=0;
		while(i<=height.length){
			int h=(i==height.length?0:height[i]);
			if(sta.isEmpty() ||h>=height[sta.peek()]){
				sta.push(i);
				i++;
			}else{
				int curHeight=height[sta.pop()];
				maxArea=Math.max(maxArea, curHeight*(sta.isEmpty()?i:i-1-sta.peek()));
			}
		}
		return maxArea;
	}
	public static void main(String[] args) {
//		char[][] ch={{'1','1','0','1'},{'1','0','1','1'},{'1','1','0','1'}};
		int[] in={5,6,4};
//		System.out.println(maximalRectangle(ch));
		System.out.println(largestRecangleArea(in));
	}
}
