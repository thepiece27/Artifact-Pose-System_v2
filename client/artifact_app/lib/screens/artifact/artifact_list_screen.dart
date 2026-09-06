import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/artifact.dart';
import '../../models/artifact_status.dart';
import '../../providers/artifact_provider.dart';
import '../../providers/auth_provider.dart';
import '../../services/api_config.dart';
import '../../theme.dart';
import '../../widgets/responsive_scaffold.dart';
import '../../widgets/status_badge.dart';
import '../../widgets/authenticated_image.dart';
import 'add_artifact_screen.dart';
import 'artifact_detail_screen.dart';

class ArtifactListScreen extends StatefulWidget {
  const ArtifactListScreen({super.key});

  @override
  State<ArtifactListScreen> createState() => _ArtifactListScreenState();
}

class _ArtifactListScreenState extends State<ArtifactListScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ArtifactProvider>().refresh();
    });
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<ArtifactProvider>();
    final auth = context.watch<AuthProvider>();
    
    // Cả Admin và Operator đều được phép thêm Artifact
    final canAdd = auth.status == AuthStatus.authenticated;

    final displayArtifacts = provider.artifacts.where((a) => a.status != ArtifactStatus.archived).toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Artifacts')),
      body: RefreshIndicator(
        onRefresh: () => context.read<ArtifactProvider>().refresh(),
        child: ResponsiveBody(
          padding: const EdgeInsets.all(16),
          child: _buildBody(displayArtifacts, provider),
        ),
      ),
      floatingActionButton: canAdd
          ? FloatingActionButton.extended(
              backgroundColor: AppColors.primary,
              foregroundColor: Colors.white,
              onPressed: () async {
                final created = await Navigator.push<bool>(
                  context,
                  MaterialPageRoute(builder: (_) => const AddArtifactScreen()),
                );
                if (created == true && mounted) {
                  await context.read<ArtifactProvider>().refresh();
                }
              },
              icon: const Icon(Icons.add),
              label: const Text('Add Artifact'),
            )
          : null,
    );
  }

  Widget _buildBody(List<Artifact> artifacts, ArtifactProvider provider) {
    if (provider.loading && artifacts.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (provider.error != null && artifacts.isEmpty) {
      return ErrorStateView(
        message: provider.error!,
        onRetry: () => context.read<ArtifactProvider>().refresh(),
      );
    }
    if (artifacts.isEmpty) {
      return const EmptyStateView(
        icon: Icons.inventory_2_outlined,
        title: 'No artifacts yet',
        subtitle: 'Tap "Add Artifact" to create one.',
      );
    }
    return ListView.separated(
      itemCount: artifacts.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) => _ArtifactCard(
        artifact: artifacts[index],
      ),
    );
  }
}

class _ArtifactCard extends StatelessWidget {
  final Artifact artifact;
  const _ArtifactCard({required this.artifact});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: () => Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ArtifactDetailScreen(artifact: artifact),
          ),
        ).then((_) => context.read<ArtifactProvider>().refresh()),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            children: [
              _Thumbnail(
                hasImage: artifact.hasImage,
                imageUrl: artifact.referenceImagePath,
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      artifact.name,
                      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        const Icon(Icons.place_outlined, size: 14, color: AppColors.textMuted),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            (artifact.location != null && artifact.location!.isNotEmpty) ? artifact.location! : '—',
                            style: const TextStyle(color: AppColors.textMuted),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    StatusBadge(status: artifact.status, compact: true),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: AppColors.textMuted),
            ],
          ),
        ),
      ),
    );
  }
}

class _Thumbnail extends StatelessWidget {
  final bool hasImage;
  final String? imageUrl;

  const _Thumbnail({required this.hasImage, required this.imageUrl});

  @override
  Widget build(BuildContext context) {
    if (!hasImage || imageUrl == null || imageUrl!.isEmpty) {
      return Container(
        width: 64, height: 64,
        decoration: BoxDecoration(color: AppColors.surfaceMuted, borderRadius: BorderRadius.circular(14)),
        child: const Icon(Icons.image_not_supported_outlined, color: AppColors.textFaint),
      );
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: AuthenticatedImage(
        ApiConfig.resolveAssetUrl(imageUrl),
        width: 64, height: 64, fit: BoxFit.cover,
        errorBuilder: (_, __, ___) => Container(
          width: 64, height: 64, color: AppColors.surfaceMuted,
          child: const Icon(Icons.broken_image_outlined, color: AppColors.textFaint),
        ),
      ),
    );
  }
}
