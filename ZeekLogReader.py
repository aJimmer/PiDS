
class ZeekLogReader(FileTailer):
    def __init__(self, filepath, delimiter='\t', tail=False, strict=False):
        # First check if the file exists and is readable
        if not os.access(filepath, os.R_OK):
            raise IOError('Could not read/access zeek log file: {:s}'.format(filepath))

        # Setup some class instance vars
        self._filepath = filepath
        self._delimiter = delimiter
        self._tail = tail
        self._strict = strict

        # Setup the Zeek to Python Type mapper
        self.field_names = []
        self.field_types = []
        self.type_converters = []
        self.type_mapper = {'bool': lambda x: True if x == 'T' else False,
                            'count': int,
                            'int': int,
                            'double': float,
                            'time': lambda x: datetime.datetime.fromtimestamp(float(x)),
                            'interval': lambda x: datetime.timedelta(seconds=float(x)),
                            'string': lambda x: x,
                            'enum': lambda x: x,
                            'port': int,
                            'unknown': lambda x: x}
        self.dash_mapper = {'bool': False, 'count': 0, 'int': 0, 'port': 0, 'double': 0.0,
                            'time': datetime.datetime.fromtimestamp(86400), 'interval': datetime.timedelta(seconds=0),
                            'string': '-', 'unknown:': '-'}

        # Initialize the Parent Class
        super(ZeekLogReader, self).__init__(self._filepath, full_read=True, tail=self._tail)

    def readrows(self):
        # Calling the internal _readrows so we can catch issues/log rotations
        reconnecting = True
        while True:
            # Yield the rows from the internal reader
            try:
                for row in self._readrows():
                    if reconnecting:
                        print('Successfully monitoring {:s}...'.format(self._filepath))
                        reconnecting = False
                    yield row
            except IOError:
                # If the tail option is set then we do a retry (might just be a log rotation)
                if self._tail:
                    print('Could not open file {:s} Retrying...'.format(self._filepath))
                    reconnecting = True
                    time.sleep(5)
                    continue
                else:
                    break

            # If the tail option is set then we do a retry (might just be a log rotation)
            if self._tail:
                print('File closed {:s} Retrying...'.format(self._filepath))
                reconnecting = True
                time.sleep(5)
                continue
            else:
                break

    def _readrows(self):
        """Internal method _readrows, see readrows() for description"""

        # Read in the Zeek Headers
        offset, self.field_names, self.field_types, self.type_converters = self._parse_zeek_header(self._filepath)

        # Use parent class to yield each row as a dictionary
        for line in self.readlines(offset=offset):

            # Check for #close
            if line.startswith('#close'):
                return

            # Yield the line as a dict
            yield self.make_dict(line.strip().split(self._delimiter))

    def _parse_zeek_header(self, zeek_log):
        """Parse the Zeek log header section.
            Format example:
                #separator \x09
                #set_separator	,
                #empty_field	(empty)
                #unset_field	-
                #path	httpheader_recon
                #fields	ts	origin	useragent	header_events_json
                #types	time	string	string	string
        """

        # Open the Zeek logfile
        with open(zeek_log, 'r') as zeek_file:

            # Skip until you find the #fields line
            _line = zeek_file.readline()
            while not _line.startswith('#fields'):
                _line = zeek_file.readline()

            # Read in the field names: ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	local_orig	local_resp	missed_bytes	history	orig_pkts	orig_ip_bytes	resp_pkts	resp_ip_bytes	tunnel_parents
            field_names = _line.strip().split(self._delimiter)[1:]

            # Read in the types: time	string	addr	port	addr	port	enum	string	interval	count	count	string	bool	bool	count	string	count	count	count	count	set[string] 
            _line = zeek_file.readline()
            field_types = _line.strip().split(self._delimiter)[1:]

            # Setup the type converters
            type_converters = []
            for field_type in field_types:
                type_converters.append(self.type_mapper.get(field_type, self.type_mapper['unknown']))

            # Keep the header offset
            offset = zeek_file.tell()

        # Return the header info
        return offset, field_names, field_types, type_converters

    def make_dict(self, field_values):
        ''' Internal method that makes sure any dictionary elements
            are properly cast into the correct types.
        '''
        data_dict = {}
        for key, value, field_type, converter in zip(self.field_names, field_values, self.field_types, self.type_converters):
            try:
                # We have to deal with the '-' based on the field_type
                data_dict[key] = self.dash_mapper.get(field_type, '-') if value == '-' else converter(value)
            except ValueError as exc:
                print('Conversion Issue for key:{:s} value:{:s}\n{:s}'.format(key, str(value), str(exc)))
                data_dict[key] = value
                if self._strict:
                    raise exc

        return data_dict
