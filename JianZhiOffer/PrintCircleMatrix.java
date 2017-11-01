package JianZhiOffer;

import java.util.ArrayList;
import java.util.List;

public class PrintCircleMatrix {
/*	题目描述
	输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
	例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 
	13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.*/
	
//解题：  把数组变成每一次循环一次，条件是rows>2*start && cols>2*start 
	public static ArrayList<Integer> printMatrix(int[][] matrix) {
		int rows=matrix.length;
		int cols=matrix[0].length;
		ArrayList<Integer> list=new ArrayList<Integer>();
		int start=0;
		while(rows>2*start && cols>2*start){
			printMatrixHelp(list,matrix,rows,cols,start);
			start++;
		}
		return list;
	}

	private static void printMatrixHelp(ArrayList<Integer> list, int[][] matrix,
			int rows, int cols, int start) {
		int endX=rows-start-1;
		int endY=cols-start-1;
		for(int i=start;i<=endY;i++){
			list.add(matrix[start][i]);
		}
		if(endX>start){
			for(int i=start+1;i<=endX;i++){
				list.add(matrix[i][endY]);
			}
		}
		if(endX>start && endY>start){
			for(int i=endY-1;i>=start;i--){
				list.add(matrix[endX][i]);
			}
		}
		if(endX>start+1 && endY>start){
			for(int i=endX-1;i>=start+1;i--){
				list.add(matrix[i][start]);
			}
		}
		
	}
	public static void main(String[] args) {
		int[][] in={{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
		List<Integer> list=printMatrix(in);
		for (Integer integer : list) {
			System.out.println(integer);
		}
	}
}
