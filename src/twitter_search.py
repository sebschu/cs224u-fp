import argparse
import datetime
import json
import urllib
import urlparse

class Twitter(object):

    search_url = 'http://search.twitter.com/search.json'

    def __init__(self, verbose=False):
        self.verbose = verbose
        super(Twitter, self).__init__()

    def search(self, query, until=None, rpp=100, max_results=None):

        results = []
        params = {
            'q': query,
            'rpp': rpp,
        }
        if until:
            params['until'] = until.strftime('%Y-%m-%d')

        if self.verbose:
            print(params)

        url = '%s?%s' % (self.search_url, urllib.urlencode(params))
        response = json.loads(urllib.urlopen(url).read())
        results.extend(response['results'])

        if len(results) >= max_results:
            return results
       
        while 'next_page' in response:
            url = self.search_url + response['next_page']
            response = json.loads(urllib.urlopen(url).read())
       
            if self.verbose:
                print('%s: %s' % (url, len(response['results'])))

            results.extend(response['results'])
            if len(results) >= max_results:
                break
        return results

    def search_last_day(self, *args, **kwargs):
        kwargs['until'] = datetime.datetime.now() - datetime.timedelta(days=1)
        return self.search(*args, **kwargs)

if __name__ == '__main__':
    f = open('positive_tweets.txt', 'a')
    parser = argparse.ArgumentParser(description='Search twitter')
    parser.add_argument('search', nargs=1)
    parser.add_argument('--rpp', dest='rpp', type=int, default=100, help='Results per page')
    parser.add_argument('-m', '--max-results', dest='max_results', type=int, default=100, help='Max results returned')
    parser.add_argument('-p', '--print-results', dest='print_results', action='store_true', help='Print the results')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Turn verbose on')
    args = parser.parse_args()

    twitter = Twitter(verbose=args.verbose)
    results = twitter.search_last_day(args.search, rpp=args.rpp, max_results=args.max_results)
    print('Found %s items' % (len(results)))
    if args.verbose:
        json.dumps(results, indent=4)
    if args.print_results:
        for result in results:
            if result['iso_language_code'] != 'en' or result['text'].startswith('RT '):
              continue;
            f.write(result['text'].encode('utf-8').strip().replace("\n", " ") + "\n")
            print('%s' % (result['text']))