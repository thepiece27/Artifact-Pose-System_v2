import 'package:flutter/material.dart';

import '../services/token_storage.dart';

/// Loads private server media with the same bearer token used by ApiClient.
/// Plain Image.network cannot attach Authorization headers, so it would fail
/// once the server's /uploads route is no longer public.
class AuthenticatedImage extends StatelessWidget {
  final String url;
  final double? width;
  final double? height;
  final BoxFit? fit;
  final ImageLoadingBuilder? loadingBuilder;
  final ImageErrorWidgetBuilder? errorBuilder;

  const AuthenticatedImage(
    this.url, {
    super.key,
    this.width,
    this.height,
    this.fit,
    this.loadingBuilder,
    this.errorBuilder,
  });

  @override
  Widget build(BuildContext context) {
    if (url.isEmpty) return const SizedBox.shrink();
    return FutureBuilder<String?>(
      future: TokenStorage().readToken(),
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return SizedBox(width: width, height: height);
        }
        final token = snapshot.data;
        final headers = token == null || token.isEmpty
            ? null
            : <String, String>{'Authorization': 'Bearer $token'};
        return Image.network(
          url,
          width: width,
          height: height,
          fit: fit,
          headers: headers,
          loadingBuilder: loadingBuilder,
          errorBuilder: errorBuilder,
        );
      },
    );
  }
}
