# Sistema de Monitoramento de Queimadas na Amazônia
# Análise de Dados Históricos de Incêndios Florestais

import pandas as pd
import numpy as np
import matplotlib

# Configurar matplotlib para não usar interface gráfica
matplotlib.use('Agg')  # Usar backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

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
                'estacao_seca': month in [5, 6, 7, 8, 9, 10]
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

        q99 = self.data['tamanho_ha'].quantile(0.99)
        outliers_removed = len(self.data[self.data['tamanho_ha'] > q99])
        self.data = self.data[self.data['tamanho_ha'] <= q99]

        self.processed_data = self.data.copy()

        print(f"Duplicatas removidas: {duplicates_removed}")
        print(f"Outliers removidos: {outliers_removed}")
        print(f"Total após limpeza: {len(self.processed_data)} registros")

    def descriptive_analysis(self):
        if self.processed_data is None:
            self.clean_and_process_data()

        total_fires = len(self.processed_data)
        total_area = self.processed_data['tamanho_ha'].sum()

        print(f"\nTotal de incêndios: {total_fires}")
        print(f"Área total queimada: {total_area:.2f} ha")

        yearly = self.processed_data.groupby('ano').agg(
            Numero_Incêndios=('data', 'count'),
            Área_Total_ha=('tamanho_ha', 'sum'),
            Tamanho_Médio_ha=('tamanho_ha', 'mean')
        )
        print("\nResumo anual:\n", yearly)

        causes = self.processed_data.groupby('causa').agg(
            Frequência=('data', 'count'),
            Área_Total_ha=('tamanho_ha', 'sum'),
            Tamanho_Médio_ha=('tamanho_ha', 'mean')
        )
        print("\nPor causa:\n", causes)

    def create_visualizations(self):
        if self.processed_data is None:
            self.clean_and_process_data()

        try:
            # Criar figura com subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Análise de Incêndios na Amazônia', fontsize=16, fontweight='bold')

            # Gráfico 1: Número de incêndios por mês
            try:
                monthly_counts = self.processed_data['mes'].value_counts().sort_index()
                axes[0, 0].bar(monthly_counts.index, monthly_counts.values, color='orange', alpha=0.7)
                axes[0, 0].set_title('Número de Incêndios por Mês')
                axes[0, 0].set_xlabel('Mês')
                axes[0, 0].set_ylabel('Quantidade')
                axes[0, 0].grid(True, alpha=0.3)
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Erro no gráfico 1:\n{str(e)[:50]}...',
                                transform=axes[0, 0].transAxes, ha='center', va='center')

            # Gráfico 2: Distribuição por causa
            try:
                cause_counts = self.processed_data['causa'].value_counts()
                axes[0, 1].pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Distribuição por Causa')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Erro no gráfico 2:\n{str(e)[:50]}...',
                                transform=axes[0, 1].transAxes, ha='center', va='center')

            # Gráfico 3: Área queimada por estado
            try:
                state_area = self.processed_data.groupby('estado')['tamanho_ha'].sum().sort_values(ascending=False)
                axes[1, 0].barh(state_area.index, state_area.values, color='red', alpha=0.6)
                axes[1, 0].set_title('Área Total Queimada por Estado (ha)')
                axes[1, 0].set_xlabel('Área (hectares)')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Erro no gráfico 3:\n{str(e)[:50]}...',
                                transform=axes[1, 0].transAxes, ha='center', va='center')

            # Gráfico 4: Evolução temporal
            try:
                temporal = self.processed_data.groupby(['ano', 'mes']).size().reset_index(name='count')
                # Criar data corretamente com dia = 1
                temporal['data'] = pd.to_datetime(temporal.assign(dia=1)[['ano', 'mes', 'dia']])
                axes[1, 1].plot(temporal['data'], temporal['count'], marker='o', linewidth=2, markersize=4)
                axes[1, 1].set_title('Evolução Temporal dos Incêndios')
                axes[1, 1].set_xlabel('Data')
                axes[1, 1].set_ylabel('Número de Incêndios')
                axes[1, 1].grid(True, alpha=0.3)

                # Rotacionar labels do eixo x para melhor legibilidade
                for tick in axes[1, 1].get_xticklabels():
                    tick.set_rotation(45)
            except Exception as e:
                # Se falhar, criar gráfico alternativo simples
                print(f"Aviso: Erro no gráfico temporal, criando versão simplificada: {e}")
                yearly_counts = self.processed_data['ano'].value_counts().sort_index()
                axes[1, 1].bar(yearly_counts.index, yearly_counts.values, color='blue', alpha=0.6)
                axes[1, 1].set_title('Incêndios por Ano')
                axes[1, 1].set_xlabel('Ano')
                axes[1, 1].set_ylabel('Número de Incêndios')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Salvar o gráfico
            try:
                plt.savefig('incendios_visualizacao.png', dpi=300, bbox_inches='tight')
                print("✅ Gráfico salvo como 'incendios_visualizacao.png'")
            except Exception as e:
                print(f"⚠️  Erro ao salvar gráfico: {e}")

            # Limpar a figura da memória
            plt.close()

        except Exception as e:
            print(f"❌ Erro geral na criação de visualizações: {e}")
            print("Continuando análise sem gráficos...")
            plt.close('all')  # Fechar todas as figuras

    def create_additional_analysis(self):
        """Análise adicional com estatísticas mais detalhadas"""
        if self.processed_data is None:
            self.clean_and_process_data()

        print("\n" + "=" * 60)
        print("ANÁLISE DETALHADA ADICIONAL")
        print("=" * 60)

        # Análise por estação seca vs úmida
        season_analysis = self.processed_data.groupby('estacao_seca').agg({
            'tamanho_ha': ['count', 'sum', 'mean', 'median'],
            'data': 'count'
        }).round(2)

        print("\n📊 Análise por Estação:")
        print("Estação Seca (Maio-Outubro) vs Úmida (Novembro-Abril)")
        print(season_analysis)

        # Top 3 estados mais afetados
        top_states = self.processed_data.groupby('estado').agg({
            'tamanho_ha': ['count', 'sum', 'mean']
        }).round(2).sort_values(('tamanho_ha', 'sum'), ascending=False).head(3)

        print("\n🔥 Top 3 Estados Mais Afetados:")
        print(top_states)

        # Análise de tamanho dos incêndios
        size_distribution = self.processed_data['classificacao_tamanho'].value_counts()
        print("\n📏 Distribuição por Tamanho:")
        for size, count in size_distribution.items():
            percentage = (count / len(self.processed_data)) * 100
            print(f"   {size}: {count} ({percentage:.1f}%)")

        # Estatísticas gerais
        print(f"\n📈 Estatísticas Gerais:")
        print(f"   Maior incêndio: {self.processed_data['tamanho_ha'].max():.2f} ha")
        print(f"   Menor incêndio: {self.processed_data['tamanho_ha'].min():.2f} ha")
        print(f"   Média: {self.processed_data['tamanho_ha'].mean():.2f} ha")
        print(f"   Mediana: {self.processed_data['tamanho_ha'].median():.2f} ha")

    def generate_recommendations(self):
        print("\n" + "=" * 60)
        print("🎯 RECOMENDAÇÕES ESTRATÉGICAS")
        print("=" * 60)

        recommendations = [
            "🔥 PREVENÇÃO: Intensificar monitoramento na estação seca (maio-outubro)",
            "🛡️  FISCALIZAÇÃO: Aumentar presença em Pará, Amazonas e Rondônia",
            "📡 TECNOLOGIA: Implementar sistemas de alerta em tempo real",
            "👥 EDUCAÇÃO: Campanhas focadas na redução de queimadas humanas (70% dos casos)",
            "🌿 RESTAURAÇÃO: Programas de recuperação de áreas degradadas",
            "🤝 PARCERIAS: Cooperação entre estados e órgãos federais",
            "💰 RECURSOS: Destinação de verbas específicas para prevenção",
            "📊 DADOS: Melhorar coleta e análise de dados em tempo real"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")

    def export_summary_report(self):
        """Gerar relatório resumido em texto"""
        if self.processed_data is None:
            self.clean_and_process_data()

        report_filename = f"relatorio_incendios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE INCÊNDIOS NA AMAZÔNIA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

            f.write(f"RESUMO EXECUTIVO:\n")
            f.write(f"- Total de incêndios analisados: {len(self.processed_data)}\n")
            f.write(f"- Área total queimada: {self.processed_data['tamanho_ha'].sum():.2f} hectares\n")
            f.write(f"- Período analisado: 2021-2024\n\n")

            # Adicionar estatísticas por ano
            yearly_stats = self.processed_data.groupby('ano').agg({
                'tamanho_ha': ['count', 'sum']
            }).round(2)

            f.write("ESTATÍSTICAS ANUAIS:\n")
            for year in yearly_stats.index:
                count = yearly_stats.loc[year, ('tamanho_ha', 'count')]
                area = yearly_stats.loc[year, ('tamanho_ha', 'sum')]
                f.write(f"  {year}: {count} incêndios, {area:.2f} ha queimados\n")

        print(f"📄 Relatório exportado como: {report_filename}")

    def run_full_analysis(self):
        """Executar análise completa"""
        print("🔥 SISTEMA DE MONITORAMENTO DE QUEIMADAS NA AMAZÔNIA")
        print("=" * 60)

        try:
            self.generate_sample_data()
            self.clean_and_process_data()
            self.descriptive_analysis()
            self.create_additional_analysis()
            self.create_visualizations()
            self.export_summary_report()
            self.generate_recommendations()

            print("\n✅ Análise completa executada com sucesso!")

        except Exception as e:
            print(f"❌ Erro durante a análise: {e}")
            print("Verifique as dependências e tente novamente.")


# Execução
if __name__ == "__main__":
    analyzer = AmazonFireAnalyzer()
    analyzer.run_full_analysis()

    # Simular dados de status atual (como seria retornado pela API)
    print("\n" + "=" * 60)
    print("📊 SIMULAÇÃO - STATUS ATUAL DO SISTEMA")
    print("=" * 60)

    current_status = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status_amazonia": "⚠️  ALERTA MÉDIO",
        "incendios_24h": 47,
        "area_afetada_24h": "1,247 hectares",
        "risco_nivel": "Médio-Alto",
        "estados_criticos": ["Pará", "Amazonas", "Rondônia"],
        "previsao_proximas_24h": "Condições favoráveis para novos focos"
    }

    print(f"🕒 Última atualização: {current_status['timestamp']}")
    print(f"📊 Status atual: {current_status['status_amazonia']}")
    print(f"🔥 Incêndios (24h): {current_status['incendios_24h']}")
    print(f"🌳 Área afetada (24h): {current_status['area_afetada_24h']}")
    print(f"⚠️  Nível de risco: {current_status['risco_nivel']}")
    print(f"🗺️  Estados em situação crítica: {', '.join(current_status['estados_criticos'])}")
    print(f"🔮 Previsão: {current_status['previsao_proximas_24h']}")

    print("\n" + "=" * 60)
    print("🎯 SISTEMA PRONTO PARA MONITORAMENTO CONTÍNUO")
    print("=" * 60)