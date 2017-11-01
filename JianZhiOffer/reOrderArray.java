package JianZhiOffer;

public class reOrderArray {
/*	题目描述
	输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
	使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组
	的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。*/	
	
//题解：选择用插入排序法，找到奇数位，然后找到前面的偶数，并且将连续的偶数进行右移；
	
	public void reOrderArray(int[] array) {
		for (int i = 0; i < array.length; i++) {
			if (array[i] % 2 == 1) {
				int temp = array[i];
				int j = i - 1;
				while (j >= 0 && array[j] % 2 == 0) {
					array[j + 1] = array[j];
					j--;
				}
				array[j + 1] = temp;
			}
		}
	}
//	插入排序；
	public static void insertSort(int[] arr){
		int j,temp;
		for(int i=0;i<arr.length;i++){
			temp=arr[i];
			j=i-1;
			while(j>=0&& arr[j]>temp){
				arr[j+1]=arr[j];
				j--;
			}
			arr[j+1]=temp;
		}
	}
	
	
	public static void main(String[] args) {
		int[] in={1,3,2,4,6,5};
		insertSort(in);
		for (int i : in) {
			System.out.println(i);
		}
	}
	
}
