package leetcodeForJianzhi;

import java.util.Stack;

/*84. Largest Rectangle in Histogram
题目介绍：
Given n non-negative integers representing the histogram's 
bar height where the width of each bar is 1, find the area of 
largest rectangle in the histogram.
举例
The largest rectangle is shown in the shaded area, which has area = 10 unit.

For example,
Given heights = [2,1,5,6,2,3],
return 10.*/
public class LargestRectangleinHistogram {
	public static int largestRectangleArea(int[] heights) {
		int lens=heights.length;
		Stack<Integer> sta=new Stack<Integer>();
		int maxArea=0;
		for(int i=0;i<lens;i++){
			int h=(i==lens?0:heights[i]);
//			int h=heights[i];
			if(sta.isEmpty() ||h >= heights[sta.peek()]){
				sta.push(i);
			}else{
				int tp=sta.pop();
				maxArea = Math.max(maxArea, heights[tp]
						* (sta.isEmpty() ? i : i - 1 - sta.peek()));
				i--;
				
			}
		}
		return maxArea;
	}
//	第二种方法：
	public static int largeRectangleArea2(int[] heights){
		int lens=heights.length;
		int maxArea=0;
		int[] sta=new int[lens+1];
		int is=-1;
		for(int i=0;i<=lens;i++){
			int h=(i==lens?0:heights[i]);
			while(is!=-1 && h<heights[sta[is]]){
				int height=heights[sta[is--]];
				int width=(is==-1)?i:i-1-sta[is];
				maxArea=Math.max(maxArea, height*width);
			}
			sta[++is]=i;
		}
		return maxArea;
	}
	public static void main(String[] args) {
		int[] in={5,6,4};
		System.out.println(largestRectangleArea(in));
	}
}
