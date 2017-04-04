/*
* @File: timer.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-04-29 17:15:00
* @Last Modified by:   kamilrocki
* @Last Modified time: 2015-05-02 13:57:14
*/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>

class Timer {

	public:

		Timer() = default;
		~Timer() = default;
		
		void start(void) {

			gettimeofday(&s, NULL); 
		}

		double end(void) {

		    struct timeval  diff_tv;
		    gettimeofday(&e, NULL); 

		    diff_tv.tv_usec = e.tv_usec - s.tv_usec;
		    diff_tv.tv_sec = e.tv_sec - s.tv_sec;

		    if (s.tv_usec > e.tv_usec) {
		        diff_tv.tv_usec += 1000000;
		        diff_tv.tv_sec--;
		    }

		    return (double) diff_tv.tv_sec + ( (double) diff_tv.tv_usec / 1000000.0);

		}

	protected:

		struct timeval s;
		struct timeval e;
};

#endif /* __TIMER_H__ */