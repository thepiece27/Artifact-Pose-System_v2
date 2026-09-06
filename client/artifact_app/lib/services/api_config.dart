class ApiConfig {
  // Supply a device/LAN URL per build with --dart-define=API_BASE_URL=... .
  // Do not bake a private LAN address into the release binary.
  static const String _defaultUrl = 'http://127.0.0.1:8000';

  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: _defaultUrl,
  );

  static Uri uri(String path, [Map<String, dynamic>? query]) {
    final normalizedBase = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    final stringQuery = query?.map(
      (k, v) => MapEntry(k, v?.toString() ?? ''),
    );
    return Uri.parse('$normalizedBase$normalizedPath').replace(
      queryParameters: stringQuery == null || stringQuery.isEmpty
          ? null
          : stringQuery,
    );
  }

  static String resolveAssetUrl(String? path) {
    if (path == null || path.isEmpty) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    final normalizedBase = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return '$normalizedBase$normalizedPath';
  }
}
