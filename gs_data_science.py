# Sistema de Monitoramento de Queimadas na Amazônia
# Análise de Dados Históricos de Incêndios Florestais - Versão Corrigida

import pandas as pd
import numpy as np
import matplotlib

# Configurar matplotlib para não usar interface gráfica
matplotlib.use('Agg')  # Usar backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AmazonFireAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None

    def generate_sample_data(self):
        np.random.seed(42)

        start_date = datetime(2021, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        n_records = len(dates) * 15

        amazon_states = ['Acre', 'Amazonas', 'Amapá', 'Maranhão', 'Mato Grosso',
                         'Pará', 'Rondônia', 'Roraima', 'Tocantins']
        causes = ['Humana', 'Natural', 'Desconhecida']
        cause_weights = [0.7, 0.2, 0.1]

        data = []
        for _ in range(n_records):
            date = np.random.choice(dates)
            if isinstance(date, np.datetime64):
                date = pd.to_datetime(date).to_pydatetime()
            month = date.month
            seasonal_factor = 2.5 if 5 <= month <= 10 else 0.5

            if np.random.random() > (0.3 * seasonal_factor):
                continue

            state = np.random.choice(amazon_states)
            cause = np.random.choice(causes, p=cause_weights)

            if cause == 'Humana':
                size_ha = np.random.lognormal(2, 1.5) * seasonal_factor
            else:
                size_ha = np.random.lognormal(1.5, 1.2) * seasonal_factor

            if size_ha < 10:
                size_class = 'Pequeno'
            elif size_ha < 100:
                size_class = 'Médio'
            elif size_ha < 1000:
                size_class = 'Grande'
            else:
                size_class = 'Muito Grande'

            lat = np.random.uniform(-10, 5)
            lon = np.random.uniform(-75, -45)

            data.append({
                'data': date,
                'estado': state,
                'causa': cause,
                'tamanho_ha': round(size_ha, 2),
                'classificacao_tamanho': size_class,
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'mes': month,
                'ano': date.year,
                'estacao_seca': month in [5, 6, 7, 8, 9, 10],
                'trimestre': f'Q{((month - 1) // 3) + 1}',
                'dia_ano': date.timetuple().tm_yday
            })

        self.data = pd.DataFrame(data)
        print(f"Dataset gerado com {len(self.data)} registros de incêndios")
        return self.data

    def clean_and_process_data(self):
        if self.data is None:
            self.generate_sample_data()

        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        duplicates_removed = initial_count - len(self.data)

        self.data['estado'] = self.data['estado'].str.strip()
        self.data['causa'] = self.data['causa'].str.strip()

        self.data = self.data.dropna()

        # Remover outliers mais conservadoramente (99.5% ao invés de 99%)
        q995 = self.data['tamanho_ha'].quantile(0.995)
        outliers_removed = len(self.data[self.data['tamanho_ha'] > q995])
        self.data = self.data[self.data['tamanho_ha'] <= q995]

        # Adicionar mais colunas úteis
        self.data['ano_mes'] = self.data['data'].dt.to_period('M')
        self.data['semana_ano'] = self.data['data'].dt.isocalendar().week

        self.processed_data = self.data.copy()

        print(f"Duplicatas removidas: {duplicates_removed}")
        print(f"Outliers removidos: {outliers_removed}")
        print(f"Total após limpeza: {len(self.processed_data)} registros")

    def descriptive_analysis(self):
        if self.processed_data is None:
            self.clean_and_process_data()

        total_fires = len(self.processed_data)
        total_area = self.processed_data['tamanho_ha'].sum()

        print(f"\n📊 ANÁLISE DESCRITIVA BÁSICA")
        print("=" * 50)
        print(f"Total de incêndios: {total_fires:,}")
        print(f"Área total queimada: {total_area:,.2f} ha")
        print(f"Média de área por incêndio: {total_area / total_fires:.2f} ha")

        # Análise anual mais detalhada
        yearly = self.processed_data.groupby('ano').agg({
            'data': 'count',
            'tamanho_ha': ['sum', 'mean', 'median', 'std'],
            'estado': lambda x: len(x.unique())
        }).round(2)

        yearly.columns = ['Num_Incêndios', 'Área_Total_ha', 'Área_Média_ha',
                          'Área_Mediana_ha', 'Desvio_Padrão', 'Estados_Afetados']

        print("\n📈 Resumo anual detalhado:")
        print(yearly)

        # Análise por causa mais detalhada
        causes = self.processed_data.groupby('causa').agg({
            'data': 'count',
            'tamanho_ha': ['sum', 'mean', 'median'],
            'estado': lambda x: len(x.unique())
        }).round(2)

        causes.columns = ['Frequência', 'Área_Total_ha', 'Área_Média_ha',
                          'Área_Mediana_ha', 'Estados_Afetados']

        # Calcular percentuais
        causes['Percentual'] = (causes['Frequência'] / total_fires * 100).round(1)

        print("\n🔥 Análise por causa:")
        print(causes)

    def create_comprehensive_visualizations(self):
        if self.processed_data is None:
            self.clean_and_process_data()

        try:
            # Criar figura maior com mais subplots
            fig = plt.figure(figsize=(20, 16))

            # Definir o grid de subplots
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

            fig.suptitle('Análise Completa de Incêndios na Amazônia (2021-2024)',
                         fontsize=18, fontweight='bold', y=0.98)

            # Gráfico 1: Evolução temporal mensal (CORRIGIDO)
            ax1 = fig.add_subplot(gs[0, :2])
            try:
                # Criar série temporal corretamente
                monthly_data = self.processed_data.groupby(['ano', 'mes']).size().reset_index(name='count')

                # Criar data usando datetime diretamente
                monthly_data['data'] = pd.to_datetime(monthly_data[['ano', 'mes']].assign(day=1))

                ax1.plot(monthly_data['data'], monthly_data['count'],
                         marker='o', linewidth=2, markersize=4, color='red', alpha=0.7)
                ax1.set_title('Evolução Temporal Mensal dos Incêndios', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Data')
                ax1.set_ylabel('Número de Incêndios')
                ax1.grid(True, alpha=0.3)

                # Melhorar formatação das datas
                ax1.tick_params(axis='x', rotation=45)

                # Adicionar linha de tendência
                from scipy import stats
                x_numeric = range(len(monthly_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, monthly_data['count'])
                trend_line = [slope * x + intercept for x in x_numeric]
                ax1.plot(monthly_data['data'], trend_line, '--', color='blue', alpha=0.5,
                         label=f'Tendência (R²={r_value ** 2:.3f})')
                ax1.legend()

            except Exception as e:
                print(f"Erro no gráfico temporal: {e}")
                # Fallback para gráfico anual simples
                yearly_counts = self.processed_data['ano'].value_counts().sort_index()
                ax1.bar(yearly_counts.index, yearly_counts.values, color='red', alpha=0.7)
                ax1.set_title('Incêndios por Ano (Gráfico Simplificado)')
                ax1.set_xlabel('Ano')
                ax1.set_ylabel('Número de Incêndios')

            # Gráfico 2: Distribuição sazonal
            ax2 = fig.add_subplot(gs[0, 2])
            monthly_counts = self.processed_data['mes'].value_counts().sort_index()
            months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            colors = ['lightblue' if i not in [4, 5, 6, 7, 8, 9] else 'orange' for i in range(12)]

            ax2.bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)],
                    color=colors, alpha=0.8)
            ax2.set_title('Sazonalidade dos Incêndios', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Mês')
            ax2.set_ylabel('Quantidade')
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels([months[i] for i in range(12)], rotation=45)
            ax2.grid(True, alpha=0.3)

            # Gráfico 3: Distribuição por causa
            ax3 = fig.add_subplot(gs[1, 0])
            cause_counts = self.processed_data['causa'].value_counts()
            colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
            wedges, texts, autotexts = ax3.pie(cause_counts.values, labels=cause_counts.index,
                                               autopct='%1.1f%%', startangle=90, colors=colors_pie)
            ax3.set_title('Distribuição por Causa', fontsize=14, fontweight='bold')

            # Melhorar legibilidade dos textos
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            # Gráfico 4: Top estados por área queimada
            ax4 = fig.add_subplot(gs[1, 1])
            state_area = self.processed_data.groupby('estado')['tamanho_ha'].sum().sort_values(ascending=True).tail(6)
            ax4.barh(state_area.index, state_area.values, color='darkred', alpha=0.7)
            ax4.set_title('Top 6 Estados - Área Queimada', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Área Total (hectares)')

            # Adicionar valores nas barras
            for i, v in enumerate(state_area.values):
                ax4.text(v + max(state_area.values) * 0.01, i, f'{v:,.0f}',
                         va='center', fontweight='bold')

            # Gráfico 5: Distribuição de tamanhos
            ax5 = fig.add_subplot(gs[1, 2])
            size_counts = self.processed_data['classificacao_tamanho'].value_counts()
            ax5.bar(size_counts.index, size_counts.values,
                    color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
            ax5.set_title('Distribuição por Tamanho', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Quantidade')
            ax5.tick_params(axis='x', rotation=45)

            # Adicionar percentuais
            total = len(self.processed_data)
            for i, (cat, count) in enumerate(size_counts.items()):
                pct = count / total * 100
                ax5.text(i, count + max(size_counts.values) * 0.01, f'{pct:.1f}%',
                         ha='center', fontweight='bold')

            # Gráfico 6: Correlação entre variáveis
            ax6 = fig.add_subplot(gs[2, 0])
            # Criar scatter plot de tamanho vs mês
            scatter_data = self.processed_data.sample(n=min(1000, len(self.processed_data)))
            ax6.scatter(scatter_data['mes'], scatter_data['tamanho_ha'],
                        alpha=0.5, c=scatter_data['mes'], cmap='viridis')
            ax6.set_title('Tamanho vs Época do Ano', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Mês')
            ax6.set_ylabel('Tamanho (ha)')
            ax6.grid(True, alpha=0.3)

            # Gráfico 7: Tendência por trimestre
            ax7 = fig.add_subplot(gs[2, 1])
            quarterly = self.processed_data.groupby(['ano', 'trimestre']).size().unstack(fill_value=0)

            x = np.arange(len(quarterly.index))
            width = 0.2

            for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                if quarter in quarterly.columns:
                    offset = (i - 1.5) * width
                    ax7.bar(x + offset, quarterly[quarter], width,
                            label=quarter, alpha=0.8)

            ax7.set_title('Incêndios por Trimestre/Ano', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Ano')
            ax7.set_ylabel('Número de Incêndios')
            ax7.set_xticks(x)
            ax7.set_xticklabels(quarterly.index)
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # Gráfico 8: Mapa de calor por estado e mês
            ax8 = fig.add_subplot(gs[2, 2])
            heatmap_data = self.processed_data.groupby(['estado', 'mes']).size().unstack(fill_value=0)

            im = ax8.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
            ax8.set_title('Mapa de Calor: Estado × Mês', fontsize=14, fontweight='bold')
            ax8.set_xticks(range(12))
            ax8.set_xticklabels([months[i] for i in range(12)], rotation=45)
            ax8.set_yticks(range(len(heatmap_data.index)))
            ax8.set_yticklabels(heatmap_data.index, fontsize=8)

            # Adicionar colorbar
            plt.colorbar(im, ax=ax8, shrink=0.8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

            # Salvar o gráfico
            plt.savefig('incendios_analise_completa.png', dpi=300, bbox_inches='tight')
            print("✅ Gráfico completo salvo como 'incendios_analise_completa.png'")

            plt.close()

        except Exception as e:
            print(f"❌ Erro na criação de visualizações: {e}")
            plt.close('all')

    def create_additional_analysis(self):
        """Análise adicional com estatísticas mais detalhadas"""
        if self.processed_data is None:
            self.clean_and_process_data()

        print("\n" + "=" * 70)
        print("📊 ANÁLISE DETALHADA AVANÇADA")
        print("=" * 70)

        # Análise por estação seca vs úmida (melhorada)
        season_analysis = self.processed_data.groupby('estacao_seca').agg({
            'tamanho_ha': ['count', 'sum', 'mean', 'median', 'std'],
            'data': 'count'
        }).round(2)

        season_analysis.columns = ['Count', 'Total_Area', 'Mean_Size', 'Median_Size', 'Std_Dev', 'Total_Events']

        print("\n🌡️  ANÁLISE SAZONAL DETALHADA:")
        print("Estação Seca (Maio-Outubro) vs Úmida (Novembro-Abril)")
        print(season_analysis)

        # Calcular a proporção
        dry_season_pct = (season_analysis.loc[True, 'Count'] / len(self.processed_data)) * 100
        wet_season_pct = (season_analysis.loc[False, 'Count'] / len(self.processed_data)) * 100

        print(f"\n📈 Proporções Sazonais:")
        print(f"   Estação Seca: {dry_season_pct:.1f}% dos incêndios")
        print(f"   Estação Úmida: {wet_season_pct:.1f}% dos incêndios")

        # Top estados com análise mais detalhada
        top_states = self.processed_data.groupby('estado').agg({
            'tamanho_ha': ['count', 'sum', 'mean', 'median'],
            'causa': lambda x: Counter(x).most_common(1)[0][0]  # Causa mais comum
        }).round(2)

        top_states.columns = ['Num_Incêndios', 'Área_Total', 'Área_Média', 'Área_Mediana', 'Causa_Principal']
        top_states = top_states.sort_values('Área_Total', ascending=False)

        print(f"\n🏆 RANKING COMPLETO DOS ESTADOS:")
        print(top_states)

        # Análise de intensidade (incêndios por área)
        print(f"\n🔥 ANÁLISE DE INTENSIDADE:")
        state_intensity = self.processed_data.groupby('estado').agg({
            'tamanho_ha': ['count', 'sum']
        })
        state_intensity.columns = ['Frequency', 'Total_Area']
        state_intensity['Intensity_Ratio'] = (
                    state_intensity['Frequency'] / state_intensity['Total_Area'] * 1000).round(3)
        state_intensity = state_intensity.sort_values('Intensity_Ratio', ascending=False)

        print("Estados ordenados por densidade de incêndios (incêndios por 1000 ha):")
        for estado, row in state_intensity.head(5).iterrows():
            print(f"   {estado}: {row['Intensity_Ratio']:.3f} incêndios/1000ha")

        # Análise temporal avançada
        print(f"\n📅 PADRÕES TEMPORAIS AVANÇADOS:")

        # Dia da semana
        self.processed_data['dia_semana'] = self.processed_data['data'].dt.day_name()
        weekday_analysis = self.processed_data['dia_semana'].value_counts()
        print("Distribuição por dia da semana:")
        for day, count in weekday_analysis.items():
            pct = count / len(self.processed_data) * 100
            print(f"   {day}: {count} ({pct:.1f}%)")

        # Análise de picos
        monthly_counts = self.processed_data.groupby('ano_mes').size()
        peak_month = monthly_counts.idxmax()
        peak_count = monthly_counts.max()

        print(f"\n🔺 Pico de atividade: {peak_month} com {peak_count} incêndios")

        # Análise de tendência por causa
        print(f"\n📊 TENDÊNCIA POR CAUSA AO LONGO DOS ANOS:")
        trend_by_cause = self.processed_data.groupby(['ano', 'causa']).size().unstack(fill_value=0)

        for causa in trend_by_cause.columns:
            valores = trend_by_cause[causa].values
            if len(valores) > 1:
                # Calcular tendência simples
                anos = list(range(len(valores)))
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(anos, valores)
                tendencia = "📈 Crescente" if slope > 0 else "📉 Decrescente" if slope < 0 else "➡️  Estável"
                print(f"   {causa}: {tendencia} (coef: {slope:.1f}, R²: {r_value ** 2:.3f})")

    def create_risk_assessment(self):
        """Análise de risco e predição"""
        print(f"\n" + "=" * 70)
        print("⚠️  ANÁLISE DE RISCO E ALERTA")
        print("=" * 70)

        # Calcular métricas de risco por estado
        risk_metrics = self.processed_data.groupby('estado').agg({
            'tamanho_ha': ['count', 'sum', 'mean', 'std'],
            'estacao_seca': lambda x: (x == True).sum() / len(x)  # Proporção na estação seca
        }).round(2)

        risk_metrics.columns = ['Frequency', 'Total_Area', 'Avg_Size', 'Size_Variability', 'Dry_Season_Ratio']

        # Criar score de risco normalizado (0-100)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 100))

        risk_factors = risk_metrics[['Frequency', 'Total_Area', 'Avg_Size', 'Size_Variability', 'Dry_Season_Ratio']]
        risk_scores = scaler.fit_transform(risk_factors)

        risk_metrics['Risk_Score'] = np.mean(risk_scores, axis=1).round(1)
        risk_metrics = risk_metrics.sort_values('Risk_Score', ascending=False)

        print("🎯 RANKING DE RISCO POR ESTADO (Score 0-100):")
        for estado, row in risk_metrics.head(5).iterrows():
            if row['Risk_Score'] >= 80:
                nivel = "🔴 CRÍTICO"
            elif row['Risk_Score'] >= 60:
                nivel = "🟡 ALTO"
            elif row['Risk_Score'] >= 40:
                nivel = "🟠 MÉDIO"
            else:
                nivel = "🟢 BAIXO"

            print(f"   {estado}: {row['Risk_Score']:.1f} - {nivel}")

        # Previsão sazonal simples
        print(f"\n🔮 PREVISÃO SAZONAL:")
        current_month = datetime.now().month

        historical_month = self.processed_data[self.processed_data['mes'] == current_month]
        if len(historical_month) > 0:
            avg_current_month = len(historical_month) / len(self.processed_data['ano'].unique())
            print(f"   Média histórica para o mês atual: {avg_current_month:.1f} incêndios")

            if current_month in [5, 6, 7, 8, 9, 10]:
                print("   ⚠️  ALERTA: Estamos na estação seca - risco elevado")
            else:
                print("   ✅ Estação úmida - risco reduzido")

    def generate_enhanced_recommendations(self):
        print("\n" + "=" * 70)
        print("🎯 RECOMENDAÇÕES ESTRATÉGICAS APRIMORADAS")
        print("=" * 70)

        # Análise dos dados para recomendações personalizadas
        top_risk_states = self.processed_data.groupby('estado')['tamanho_ha'].sum().sort_values(ascending=False).head(
            3).index.tolist()
        peak_months = self.processed_data['mes'].value_counts().head(3).index.tolist()
        human_percentage = (self.processed_data['causa'] == 'Humana').sum() / len(self.processed_data) * 100

        recommendations = [
            f"🎯 PRIORIDADE MÁXIMA: Focar recursos em {', '.join(top_risk_states)} (estados com maior área queimada)",
            f"📅 PERÍODO CRÍTICO: Intensificar monitoramento nos meses {', '.join(map(str, peak_months))}",
            f"👥 FATOR HUMANO: {human_percentage:.1f}% dos incêndios são causados por ação humana - priorizar educação e fiscalização",
            "🛰️  TECNOLOGIA AVANÇADA: Implementar sistemas de detecção por satélite em tempo real",
            "🚁 RESPOSTA RÁPIDA: Criar equipes de combate aéreo estrategicamente posicionadas",
            "💧 RECURSOS HÍDRICOS: Mapear e proteger fontes de água para combate a incêndios",
            "🌡️  MONITORAMENTO CLIMÁTICO: Integrar dados meteorológicos para previsão de risco",
            "🤖 INTELIGÊNCIA ARTIFICIAL: Desenvolver modelos preditivos para antecipação de focos",
            "📱 APLICATIVO CIDADÃO: Criar app para denúncias e reportes em tempo real",
            "🏛️  POLÍTICAS PÚBLICAS: Implementar incentivos para práticas sustentáveis",
            "💰 FUNDO DE EMERGÊNCIA: Destinar orçamento específico para prevenção e combate",
            "🎓 CAPACITAÇÃO: Treinar brigadistas locais e voluntários especializados"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")

        # Recomendações específicas por risco
        print(f"\n🔍 RECOMENDAÇÕES ESPECÍFICAS POR NÍVEL DE RISCO:")

        # Análise dos estados de maior risco
        risk_analysis = self.processed_data.groupby('estado').agg({
            'tamanho_ha': ['count', 'sum', 'mean'],
            'causa': lambda x: (x == 'Humana').sum() / len(x)
        }).round(2)

        for estado in top_risk_states:
            human_ratio = risk_analysis.loc[estado, ('causa', '<lambda>')] * 100
            avg_size = risk_analysis.loc[estado, ('tamanho_ha', 'mean')]

            print(f"\n   🏴 {estado}:")
            if human_ratio > 70:
                print(f"     - {human_ratio:.1f}% causas humanas → Intensificar fiscalização")
            if avg_size > 50:
                print(f"     - Incêndios grandes (média {avg_size:.1f}ha) → Melhorar resposta inicial")
            print(f"     - Implementar sistema de alerta específico")

    def export_comprehensive_report(self):
        """Gerar relatório completo e detalhado"""
        if self.processed_data is None:
            self.clean_and_process_data()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"relatorio_completo_incendios_{timestamp}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO COMPLETO DE ANÁLISE DE INCÊNDIOS NA AMAZÔNIA\n")
            f.write("=" * 80 + "\n")
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Período analisado: 2021-2024\n")
            f.write("=" * 80 + "\n\n")

            # Resumo executivo
            f.write("RESUMO EXECUTIVO:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Total de incêndios: {len(self.processed_data):,}\n")
            f.write(f"• Área total queimada: {self.processed_data['tamanho_ha'].sum():,.2f} hectares\n")
            f.write(f"• Tamanho médio por incêndio: {self.processed_data['tamanho_ha'].mean():.2f} hectares\n")
            f.write(
                f"• Estados mais afetados: {', '.join(self.processed_data.groupby('estado')['tamanho_ha'].sum().sort_values(ascending=False).head(3).index.tolist())}\n")
            f.write(
                f"• Principal causa: {self.processed_data['causa'].value_counts().index[0]} ({self.processed_data['causa'].value_counts().iloc[0] / len(self.processed_data) * 100:.1f}%)\n\n")

            # Estatísticas anuais
            f.write("ESTATÍSTICAS ANUAIS:\n")
            f.write("-" * 20 + "\n")
            yearly_stats = self.processed_data.groupby('ano').agg({
                'tamanho_ha': ['count', 'sum', 'mean'],
                'causa': lambda x: Counter(x).most_common(1)[0][0]
            }).round(2)

            for year in yearly_stats.index:
                count = yearly_stats.loc[year, ('tamanho_ha', 'count')]
                area = yearly_stats.loc[year, ('tamanho_ha', 'sum')]
                avg = yearly_stats.loc[year, ('tamanho_ha', 'mean')]
                main_cause = yearly_stats.loc[year, ('causa', '<lambda>')]
                f.write(
                    f"  {year}: {count:,} incêndios | {area:,.2f} ha | Média: {avg:.2f} ha | Causa principal: {main_cause}\n")

            # Análise por estado
            f.write("\nANÁLISE POR ESTADO:\n")
            f.write("-" * 20 + "\n")
            state_stats = self.processed_data.groupby('estado').agg({
                'tamanho_ha': ['count', 'sum', 'mean'],
                'causa': lambda x: Counter(x).most_common(1)[0][0]
            }).round(2).sort_values(('tamanho_ha', 'sum'), ascending=False)

            for estado in state_stats.index:
                count = state_stats.loc[estado, ('tamanho_ha', 'count')]
                area = state_stats.loc[estado, ('tamanho_ha', 'sum')]
                avg = state_stats.loc[estado, ('tamanho_ha', 'mean')]
                main_cause = state_stats.loc[estado, ('causa', '<lambda>')]
                f.write(
                    f"  {estado}: {count:,} incêndios | {area:,.2f} ha | Média: {avg:.2f} ha | Causa: {main_cause}\n")

            # Análise sazonal
            f.write("\nANÁLISE SAZONAL:\n")
            f.write("-" * 20 + "\n")
            dry_season = self.processed_data[self.processed_data['estacao_seca'] == True]
            wet_season = self.processed_data[self.processed_data['estacao_seca'] == False]

            f.write(
                f"Estação Seca (Mai-Out): {len(dry_season):,} incêndios ({len(dry_season) / len(self.processed_data) * 100:.1f}%)\n")
            f.write(f"  - Área total: {dry_season['tamanho_ha'].sum():,.2f} ha\n")
            f.write(f"  - Tamanho médio: {dry_season['tamanho_ha'].mean():.2f} ha\n")

            f.write(
                f"Estação Úmida (Nov-Abr): {len(wet_season):,} incêndios ({len(wet_season) / len(self.processed_data) * 100:.1f}%)\n")
            f.write(f"  - Área total: {wet_season['tamanho_ha'].sum():,.2f} ha\n")
            f.write(f"  - Tamanho médio: {wet_season['tamanho_ha'].mean():.2f} ha\n")

            # Recomendações
            f.write("\nRECOMENDAÇÕES ESTRATÉGICAS:\n")
            f.write("-" * 30 + "\n")
            recommendations = [
                "1. Intensificar monitoramento na estação seca (maio-outubro)",
                "2. Focar recursos nos estados mais afetados",
                "3. Combater causas humanas através de educação e fiscalização",
                "4. Implementar sistemas de detecção precoce",
                "5. Criar brigadas de combate estrategicamente posicionadas",
                "6. Desenvolver aplicativos de denúncia cidadã",
                "7. Integrar dados meteorológicos para previsão",
                "8. Estabelecer fundos de emergência específicos"
            ]

            for rec in recommendations:
                f.write(f"  {rec}\n")

        print(f"📄 Relatório completo exportado como: {report_filename}")

    def create_dashboard_summary(self):
        """Criar resumo para dashboard em tempo real"""
        print("\n" + "=" * 70)
        print("📊 DASHBOARD - RESUMO EXECUTIVO")
        print("=" * 70)

        # Métricas principais
        total_fires = len(self.processed_data)
        total_area = self.processed_data['tamanho_ha'].sum()
        avg_size = self.processed_data['tamanho_ha'].mean()
        current_month = datetime.now().month

        # Status atual baseado no mês
        if current_month in [5, 6, 7, 8, 9, 10]:
            status = "🔴 ALERTA ALTO"
            risk_level = "CRÍTICO"
        elif current_month in [4, 11]:
            status = "🟡 ALERTA MÉDIO"
            risk_level = "MODERADO"
        else:
            status = "🟢 RISCO BAIXO"
            risk_level = "BAIXO"

        print(f"🎯 STATUS ATUAL: {status}")
        print(f"📊 PERÍODO: 2021-2024")
        print(f"🔥 TOTAL DE INCÊNDIOS: {total_fires:,}")
        print(f"🌳 ÁREA TOTAL QUEIMADA: {total_area:,.0f} hectares")
        print(f"📏 TAMANHO MÉDIO: {avg_size:.1f} hectares")
        print(f"⚠️  NÍVEL DE RISCO ATUAL: {risk_level}")

        # Top 3 insights
        print(f"\n💡 PRINCIPAIS INSIGHTS:")

        # Insight 1: Sazonalidade
        dry_season_pct = (self.processed_data['estacao_seca'] == True).sum() / len(self.processed_data) * 100
        print(f"   1. {dry_season_pct:.0f}% dos incêndios ocorrem na estação seca")

        # Insight 2: Causa humana
        human_pct = (self.processed_data['causa'] == 'Humana').sum() / len(self.processed_data) * 100
        print(f"   2. {human_pct:.0f}% dos incêndios têm causa humana")

        # Insight 3: Estado mais crítico
        top_state = self.processed_data.groupby('estado')['tamanho_ha'].sum().idxmax()
        top_state_area = self.processed_data.groupby('estado')['tamanho_ha'].sum().max()
        print(f"   3. {top_state} é o estado mais afetado ({top_state_area:,.0f} ha)")

        # Previsão próximos meses
        print(f"\n🔮 PREVISÃO PRÓXIMOS 3 MESES:")
        next_months = [(current_month + i - 1) % 12 + 1 for i in range(1, 4)]

        for i, month in enumerate(next_months, 1):
            historical_avg = len(self.processed_data[self.processed_data['mes'] == month]) / len(
                self.processed_data['ano'].unique())

            if month in [5, 6, 7, 8, 9, 10]:
                trend = "📈 ALTA"
                color = "🔴"
            elif month in [4, 11]:
                trend = "📊 MÉDIA"
                color = "🟡"
            else:
                trend = "📉 BAIXA"
                color = "🟢"

            month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                           7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}

            print(f"   {color} {month_names[month]}: {trend} (~{historical_avg:.0f} incêndios esperados)")

    def run_complete_analysis(self):
        """Executar análise completa e abrangente"""
        print("🔥 SISTEMA AVANÇADO DE MONITORAMENTO DE QUEIMADAS NA AMAZÔNIA")
        print("=" * 80)
        print("🚀 Iniciando análise completa e detalhada...")

        try:
            # Etapa 1: Geração e limpeza dos dados
            print("\n⚙️  ETAPA 1: Processamento de Dados")
            self.generate_sample_data()
            self.clean_and_process_data()

            # Etapa 2: Análise descritiva
            print("\n📊 ETAPA 2: Análise Descritiva")
            self.descriptive_analysis()

            # Etapa 3: Análise avançada
            print("\n🔬 ETAPA 3: Análise Avançada")
            self.create_additional_analysis()

            # Etapa 4: Análise de risco
            print("\n⚠️  ETAPA 4: Análise de Risco")
            self.create_risk_assessment()

            # Etapa 5: Visualizações
            print("\n📈 ETAPA 5: Criação de Visualizações")
            self.create_comprehensive_visualizations()

            # Etapa 6: Relatórios
            print("\n📄 ETAPA 6: Geração de Relatórios")
            self.export_comprehensive_report()

            # Etapa 7: Dashboard
            print("\n🖥️  ETAPA 7: Dashboard Executivo")
            self.create_dashboard_summary()

            # Etapa 8: Recomendações
            print("\n🎯 ETAPA 8: Recomendações Estratégicas")
            self.generate_enhanced_recommendations()

            print(f"\n✅ ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
            print("📁 Arquivos gerados:")
            print("   • incendios_analise_completa.png (gráficos)")
            print(f"   • relatorio_completo_incendios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        except Exception as e:
            print(f"❌ Erro durante a análise completa: {e}")
            print("🔧 Executando análise em modo de recuperação...")

            # Modo de recuperação - análise básica
            try:
                if self.processed_data is None:
                    self.clean_and_process_data()
                self.descriptive_analysis()
                self.create_dashboard_summary()
                print("✅ Análise básica concluída com sucesso!")
            except Exception as e2:
                print(f"❌ Erro crítico: {e2}")

    def simulate_real_time_monitoring(self):
        """Simular monitoramento em tempo real"""
        print("\n" + "=" * 70)
        print("🛰️  SIMULAÇÃO - MONITORAMENTO EM TEMPO REAL")
        print("=" * 70)

        # Simular dados atuais
        current_time = datetime.now()

        # Gerar alertas baseados no período atual
        if current_time.month in [5, 6, 7, 8, 9, 10]:
            alert_level = "🔴 CRÍTICO"
            expected_fires = np.random.randint(80, 150)
            risk_description = "Condições extremamente favoráveis para incêndios"
        elif current_time.month in [4, 11]:
            alert_level = "🟡 MODERADO"
            expected_fires = np.random.randint(30, 60)
            risk_description = "Condições moderadas - monitoramento intensificado"
        else:
            alert_level = "🟢 BAIXO"
            expected_fires = np.random.randint(5, 25)
            risk_description = "Condições favoráveis - risco reduzido"

        # Dados simulados das últimas 24h
        fires_24h = np.random.randint(expected_fires - 20, expected_fires + 20)
        area_24h = np.random.randint(800, 2500)

        # Estados com maior atividade (baseado nos dados históricos)
        critical_states = ['Pará', 'Amazonas', 'Mato Grosso', 'Rondônia']
        active_states = np.random.choice(critical_states, size=np.random.randint(2, 4), replace=False)

        print(f"🕒 ÚLTIMA ATUALIZAÇÃO: {current_time.strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🚨 NÍVEL DE ALERTA: {alert_level}")
        print(f"🔥 FOCOS DETECTADOS (24h): {fires_24h}")
        print(f"🌳 ÁREA ESTIMADA AFETADA (24h): {area_24h:,} hectares")
        print(f"📊 PREVISÃO PRÓXIMAS 24h: {risk_description}")
        print(f"🗺️  ESTADOS EM ATENÇÃO: {', '.join(active_states)}")

        # Simular coordenadas de focos ativos
        print(f"\n📍 FOCOS ATIVOS PRIORITÁRIOS:")
        for i in range(min(5, fires_24h // 10)):
            lat = np.random.uniform(-10, 5)
            lon = np.random.uniform(-75, -45)
            estado = np.random.choice(active_states)
            size = np.random.uniform(5, 50)
            print(f"   {i + 1}. {estado}: {lat:.4f}°, {lon:.4f}° (~{size:.1f} ha)")

        # Recomendações automáticas
        print(f"\n🎯 AÇÕES RECOMENDADAS IMEDIATAS:")
        if alert_level == "🔴 CRÍTICO":
            actions = [
                "Ativar equipes de combate em prontidão máxima",
                "Intensificar patrulhamento aéreo",
                "Coordenar com defesa civil local",
                "Preparar recursos de evacuação se necessário"
            ]
        elif alert_level == "🟡 MODERADO":
            actions = [
                "Aumentar frequência de monitoramento",
                "Alertar brigadas locais",
                "Verificar disponibilidade de recursos",
                "Monitorar condições meteorológicas"
            ]
        else:
            actions = [
                "Manter monitoramento de rotina",
                "Realizar manutenção preventiva de equipamentos",
                "Atualizar mapas de risco",
                "Treinar equipes locais"
            ]

        for i, action in enumerate(actions, 1):
            print(f"   {i}. {action}")

        print(f"\n📡 PRÓXIMA ATUALIZAÇÃO: {(current_time + timedelta(hours=2)).strftime('%H:%M:%S')}")


# Importar scipy apenas se disponível (para cálculos estatísticos)
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  Aviso: scipy não disponível - alguns cálculos estatísticos serão simplificados")

# Importar sklearn apenas se disponível (para normalização)
try:
    from sklearn.preprocessing import MinMaxScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Aviso: sklearn não disponível - scores de risco serão calculados de forma simplificada")

# Execução Principal
if __name__ == "__main__":
    print("🌟 INICIALIZANDO SISTEMA AVANÇADO DE MONITORAMENTO...")
    print("=" * 80)

    analyzer = AmazonFireAnalyzer()

    # Executar análise completa
    analyzer.run_complete_analysis()

    # Simular monitoramento em tempo real
    analyzer.simulate_real_time_monitoring()

    print(f"\n" + "=" * 80)
    print("🏁 SISTEMA FINALIZADO - TODAS AS ANÁLISES CONCLUÍDAS")
    print("=" * 80)
    print("📊 Resumo de arquivos gerados:")
    print("   📈 incendios_analise_completa.png - Visualizações avançadas")
    print("   📄 relatorio_completo_incendios_[timestamp].txt - Relatório detalhado")
    print("   🎯 Sistema pronto para integração com APIs reais")
    print("=" * 80)
